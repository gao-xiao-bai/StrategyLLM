
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
import replicate
from together import Together
from typing import List, Tuple, Dict, Any

# The following are packages/funtions for exponential backoff
# (ref. https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff)
# in order to deal with OpenAI API "rate limit reached" errors
from tenacity import (
	retry,
	stop_after_attempt,
	wait_random_exponential,
)


class TimeoutError(Exception):
	pass


def log_retry(state):
	print(f"Retrying: {state.attempt_number}...")


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
	def decorator(func):
		def _handle_timeout(signum, frame):
			raise TimeoutError(error_message)

		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			signal.signal(signal.SIGALRM, _handle_timeout)
			signal.alarm(seconds)
			try:
				result = func(*args, **kwargs)
			finally:
				signal.alarm(0)
			return result
		return wrapper
	return decorator


class Base:
	def __init__(self, config: Config):
		
		# dataset parameters
		self.dataset_name = config.dataset_name

		# core parameters
		self.LM = config.LM
		self.prompt_name = config.prompt_name
		self.max_tokens = config.max_tokens
		self.output_dir = config.output_dir
		self.subset = config.subset if hasattr(config, "subset") else None
		self.task_definition = config.task_definition if hasattr(config, "task_definition") else None

		# decoding parameters
		self.n_votes = config.n_votes  # number of completions
		self.temperature = config.temperature  # temperature for the solver LM
		self.batch_size = config.batch_size  # batch size for querying the LM

		# load the API keys
		self.api_keys = itertools.cycle(config.api_keys)

	@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(30), after=log_retry)
	def _query(self, prompt: str, stop: str, LM: str, n: int = 1, temperature: float = 0.0, max_tokens: int = 1024) -> List[str]:
		"""Query the language model to generate completions.

		Args:
			prompt (str): The prompt to query the language model.
			stop (str): The stop sequence to stop the generation.
			LM (str): The language model to query.
			n (int): The number of completions to generate.
			temperature (float): The temperature for sampling.
			max_tokens (int): The maximum number of tokens to generate.

		Returns:
			list: A list of completions.
		"""
		api_key = next(self.api_keys)
		openai.api_key = api_key

		if LM in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0125", "gpt-4", "gpt-4-0613", "gpt-4-32k-0613", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13"]:
			response = openai.ChatCompletion.create(
				model=LM,
				messages=[
					{"role": "user", "content": prompt},
				],
				temperature=temperature,
				n=n,
				frequency_penalty=0,
				presence_penalty=0,
				stop=stop
			)
			choices = response["choices"]
			completion_objs = [choice.message for choice in choices]
			completions = [completion.content for completion in completion_objs]
		elif LM in ["meta/meta-llama-3-8b-instruct", "meta/meta-llama-3-70b-instruct"]:
			response = replicate.run(
				LM,
				input={
					"prompt": prompt,
					"temperature": 0.01 if temperature == 0 else temperature,
					"max_new_tokens": max_tokens,
				}
			)
			response = "".join(response)
			completions = [response]
		elif LM in ["meta-llama/Llama-3-8b-chat-hf", "meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x22B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
			together_client = Together()
			response = together_client.chat.completions.create(
				model=LM,
				messages=[{"role": "user", "content": prompt}],
				temperature=temperature,
				max_tokens=max_tokens,
				n=n,
			)
			choices = response.choices
			completion_objs = [choice.message for choice in choices]
			completions = [completion.content for completion in completion_objs]
		else:
			raise NotImplementedError(f"Model {LM} is not supported.")
		return completions

	