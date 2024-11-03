import os
cwd = os.getcwd()
if cwd.endswith("source/model"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from dataset.utils import MAX_TOKEN, extract_answer
import sys
import openai
import itertools
import errno
import os
import json
import requests
import signal
import functools
import re
import string
import json
import copy
from model.base import Base
from typing import List, Tuple


class SolutionMaster(Base):
	def __init__(self, config: Config):
		super(SolutionMaster, self).__init__(config)

		# load the prompt and template
		examples_path = os.path.join(config.examples_dir, f"standard_prompt_{config.few_shots}.txt")
		with open(examples_path, 'r', encoding='utf-8') as fr:
			self.examples = fr.read()

		examples = self.examples.split("\n\n\n\n")
		self.examples_list = copy.deepcopy(examples)
		print("examples: ", examples)
		for i, example in enumerate(examples):
			examples[i] = f"Example {i}:\n{example}"
		self.examples = "\n\n".join(examples)

	def write_solution(self) -> List[str]:
		"""Write solutions to the task examples.

		Returns:
			list: A list of solutions.
		"""
		results = []
		accuracy = []
		for example in self.examples_list:
			prompt = f"""
Task:
{self.task_definition}
				
Example of the task:
{example}

Please write a solution to the provided example. The answer obtained from the solution must be the same as the original answer.
The result must be in the following format:
Question: Question in the provided example
Solution: Solution to the question
Answer: Answer in the provided example, which must include the string 'The answer is '
"""
			print(f'\n****SOLUTION LLM PROMPT****\n{prompt}\n')
			result = self._query(prompt=prompt,
									  n=1,
									  stop=None,
									  max_tokens=1000,
									  LM=self.LM,
									  temperature=0)[0].strip()
			print(f'\n****SOLUTION LLM RESPONSE****\n{result}\n')

			predict_answer = extract_answer(result, self.dataset_name)
			gold_answer = extract_answer(example, self.dataset_name)
			if predict_answer == gold_answer:
				print("The solution is correct!\n")
				accuracy.append(1)
			else:
				print("The solution is incorrect!\n")
				accuracy.append(0)
			results.append(result)

		print("accuracy: ", sum(accuracy) / len(accuracy))

		with open(os.path.join(self.output_dir, f"results.txt"), "w") as f:
			f.write("\n\n\n\n".join(results))

		with open(os.path.join(self.output_dir, f"results.json"), "w") as f:
			json.dump(results, f, indent=4)

		return results

	def run(self) -> List[str]:
		"""Run the solution master.

		Returns:
			list: A list of solutions.
		"""
		results = self.write_solution()

		return results













