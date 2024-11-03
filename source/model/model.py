import os
cwd = os.getcwd()
if cwd.endswith("source/model"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from dataset.utils import MAX_TOKEN, extract_math_answer, normalize
import sys
import openai
import itertools
import errno
import os
import json
import signal
import functools
import requests
import re
import string
import time
from model.base import Base
from typing import List, Tuple, Dict, Any


class Model(Base):
	def __init__(self, config: Config):
		super(Model, self).__init__(config)

		# load the prompt and template
		examples_path = os.path.join(config.examples_dir, f"{self.prompt_name}_prompt_{config.few_shots}.txt")
		template_path = os.path.join(config.template_dir, f"{self.prompt_name}_template.txt")

		with open(examples_path, 'r', encoding='utf-8') as fr:
			self.prompt = fr.read()
			print("prompt:\n", self.prompt)

		with open(template_path, 'r', encoding='utf-8') as fr:
			self.template = fr.read()

	def predict(self, example_dict: Dict[str, Any], completion_only: bool = False) -> Dict[str, Any]:
		"""Predict the answer to a question.

		Args:
			example_dict (dict): the example dict, which is in the format:
				{
					"question": the question to be answered,
					...
				}
			completion_only (bool): if True, only return the completions, but not the answer
		
		Returns:
			dict: the output dict, which is in the format:
				{
					"answer": the answer to the question,
					"initial_answers": the initial answers,
					"answers": the answers after removing the invalid ones,
					"completion": the completion that results in the final answer,
					"completions": the list of completions
				}
		"""
		question = example_dict["question"]

		# apply the template to the question
		templated_example = self._apply_template(template=self.template, example=example_dict)

		# concatenate the few-shot prompt and the example
		prompt_and_example = f"{self.prompt}\n\n\n\n{templated_example}"

		# set the stop token
		stop_token = "Question:"

		# get the max token for the current dataset
		if self.max_tokens:  # if max_tokens is specified, use it
			max_token = self.max_tokens
		else:  # otherwise, use the default max_tokens for the current dataset
			max_token = self.get_max_token()

		# query the LM to get the completions
		n_iters = self.n_votes // self.batch_size  # number of iterations to query the LM
		completions = []
		for iter_id in range(n_iters):
			new_completions = self._query(
				prompt=prompt_and_example,
				n=self.batch_size,
				stop=[stop_token],
				max_tokens=max_token,
				LM=self.LM,
				temperature=self.temperature
			)
			completions += new_completions

		if completion_only:  # return only the completions, but not the answer
			output = {
				"answer": "",
				"initial_answers": "",
				"answers": "",
				"completion": "",
				"completions": completions
			}
			return output

		answer, final_completion, answers, initial_answers = self.derive_answer_from_completions(example=example_dict, completions=completions)

		output = {
			"answer": answer,
			"initial_answers": initial_answers,
			"answers": answers,
			"completion": final_completion,
			"completions": completions
		}

		return output

	def _apply_template(self, template: str, example: dict) -> str:
		"""Apply the template to the example.

		Args:
			template (str): the template
			example (dict): the example dict
		
		Returns:
			str: the example with the template applied
		"""
		# for every [{FIELD}] in the template, replace it with the corresponding value of the key "{field}" in the example dict
		example_in_template = template
		for field in re.findall(r"\[.*?\]", template):
			field_name = field[1:-1]
			field_name = field_name.lower()
			if field_name in example:
				example_in_template = example_in_template.replace(field, str(example[field_name]))
		return example_in_template

	def get_max_token(self) -> int:
		"""Get the max token for the current dataset.

		Returns:
			int: the max token
		"""
		max_token_dict = MAX_TOKEN
		return max_token_dict[self.dataset_name]

	def _execute(self, example: dict, completion: str) -> Any:
		"""Extract the answer from the completion.

		Args:
			example (dict): the example
			completion (str): the completion
		
		Returns:
			Any: the answer
		"""
		
		if self.dataset_name in ["LLC", "WS"]:
			if "answer is " not in completion:
				answer = "[invalid]"
			else:
				answer = normalize(completion.split("answer is ")[-1])
		elif self.dataset_name in ["MATH", "MA"]:
			answer = extract_math_answer(completion)
		elif self.dataset_name in ["DU"]:
			if "answer is " not in completion:
				answer = "[invalid]"
			else:
				answer = completion.split("answer is ")[-1].strip()
				answer = re.sub(pattern="[\s\.#]", repl="", string=answer)
				answer = answer.split("\n")[-1]  # only get the last output
				answer = answer.rstrip("Y")  # strip the trailing "Y"s if it exists
		elif self.dataset_name in ["StrategyQA"]:
			if "(yes or no)" in completion:
				completion = completion.replace("(yes or no)", "")
				completion = " ".join(completion.split())
			if "answer is" not in completion:
				if "Answer:" in completion:
					completion = completion.split("Answer:")[-1].strip()
				completion = completion.lower()
				if " yes" in completion or "yes," in completion or "yes." in completion:
					answer = True
				elif " no" in completion or "no," in completion or "no." in completion:
					answer = False
				else:
					answer = "[invalid]"
			else:
				answer = completion.split("answer is ")[-1].split()[0].strip("\n.").lower()
				if answer == "yes":
					answer = True
				elif answer == "no":
					answer = False
				elif " yes" in answer or "yes," in answer or "yes." in completion:
					answer = True
				elif " no" in answer or "no," in answer or "no." in completion:
					answer = False
				else:
					answer = "[invalid]"
		else:
			assert False, f"Execution for the dataset {self.dataset_name} is not implemented!"
		return answer

	def derive_answer_from_completions(self, example: Dict[str, Any], completions: List[str]) -> Tuple[str, str, List[str], List[str]]:
		"""Derive the answer from the completions.

		Args:
			example (dict): the example
			completions (list): the list of completions
		
		Returns:
			Tuple: a tuple of the answer, the final completion, the answers, and the initial answers
		"""

		completion_lists = {}  # a dict of lists of completions; each item is {answer: [completions that result in the same answer after execution]}
		initial_answers = []
		answers = []
		for completion in completions:
			try:
				answer = self._execute(example=example, completion=completion)  # execute the completion
			except Exception as e:
				print(f"Error executing completion: {completion}.\n Error: {e}")
				continue

			initial_answers.append(answer)

			if type(answer) == str and "invalid" in answer:
				continue

			answers.append(answer)

			# check for answer equivalence
			equivalent_found = False
			for existing_answer in completion_lists.keys():
				if existing_answer == answer:  # if the answer is equivalent to an existing answer
					completion_lists[existing_answer].append(completion)  # add the completion to list of completions corresponding to the existing answer
					equivalent_found = True
					break
			if not equivalent_found:  # if the answer is not equivalent to any existing answer
				completion_lists[answer] = [completion]  # create a new list of completions corresponding to the answer

		# get the top-voted answer as the final answer
		if len(completion_lists) == 0:  # if no valid completion is found
			return "[invalid]", completions[0], ["[invalid]"], ["[invalid]"]

		completion_lists = sorted(completion_lists.items(), key=lambda x: len(x[1]), reverse=True)  # vote for the majority answer
		final_completion = completion_lists[0][1][0]
		answer = completion_lists[0][0]

		return answer, final_completion, answers, initial_answers