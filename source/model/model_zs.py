import os
import random

cwd = os.getcwd()
if cwd.endswith("source/model"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from dataset.utils import MAX_TOKEN, extract_math_answer
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
import time
from model.model import Model
from typing import List, Tuple, Dict, Any


class ZSModel(Model):
	def __init__(self, config: Config):
		super(ZSModel, self).__init__()
		self.prompt = f"Please determine the final answer for the question based on the candidate solutions and their corresponding answers. The output should be in the following format:\nFinal Answer: The final answer, which must include the string 'The final answer is '."

	def predict(self, example_dict: Dict[str, Any]) -> Dict[str, Any]:
		"""Predict the answer to a question.

		Args:
			example_dict (dict): the example dict, which is in the format:
				{
					"question": the question to be answered,
					"candidate_solutions": the candidate solutions,
					...
				}
		
		Returns:
			dict: the output dict, which is in the format:
				{
					"answer": the answer to the question,
					"completion": the completion that results in the final answer
				}
		"""
		question = example_dict["question"]
		candidate_solutions = example_dict["candidate_solutions"]

		prompt = f"Question: {question}\n\nCandidate Solutions:\n{candidate_solutions}\n\n{self.prompt}"
		print("Prompt:\n", prompt)

		completion = self._query(
			prompt=prompt,
			n=self.batch_size,
			stop=None,
			LM=self.LM,
			temperature=self.temperature
		)[0]

		answer = self.derive_answer_from_completion(example=example_dict, completion=completion)

		output = {
			"answer": answer,
			"completion": completion,
		}
		return output

	def derive_answer_from_completion(self, example: Dict[str, Any], completion: str) -> str:
		'''Derive the answer from the completion.

		Args:
			example (dict): the example dict
			completion (str): the completion from the language model

		Returns:
			str: the answer
		'''
		
		try:
			answer = self._execute(example=example, completion=completion)
		except Exception as e:
			answer = "[error]"
			print(f"Error executing completion: {completion}.\n Error: {e}")

		return answer