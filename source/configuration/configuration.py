'''
Model configuration.
'''

import copy
import json
import os
from typing import Dict, Any


class Config(object):
	def __init__(self, **kwargs):
		# dataset parameters
		self.dataset_name = None # name of evaluation dataset
		self.split = None # split of evaluation dataset
		self.task_definition = None # task definition
		self.few_shots = None # number of few-shot examples
		self.subset = None # subset of the dataset
		self.examples_dir = None # directory containing examples
		self.template_dir = None # directory containing templates
		self.output_dir = None # directory to save output

		# strategyllm parameters
		self.max_iterations = 3 # maximum number of iterations for strategyllm
		self.threshold = 0.75 # threshold for strategyllm

		# core parameters
		self.prompt_name = kwargs.get('prompt_name', None) # name of the prompt; should be one of the names from the `prompt/` folder, e.g. "subqeustion_dependency"
		self.LM = kwargs.get('LM', "gpt-3.5-turbo") # the underlying LM
		self.max_tokens = kwargs.get('max_tokens', None) # maximum number of tokens to generate

		# decoding parameters
		self.n_votes = kwargs.get('n_votes', 1)  # number of reasoning chains to generate; default to 1 (greedy decoding)
		self.temperature = kwargs.get('temperature', 0.0)  # temperature for the LM ; default to 0.0 (greedy decoding)
		self.batch_size = 1  # number of examples to query the LM at a time

		# API keys; default to empty
		self.api_keys = []
		self.org_ids = []

	@classmethod
	def from_dict(cls, dict_obj: Dict[str, Any]) -> 'Config':
		"""Creates a Config object from a dictionary.

		Args:
			dict_obj (Dict[str, Any]): A dictionary containing the configuration parameters.

		Returns:
			config (Config): The configuration object.
		"""
		config = cls()
		for k, v in dict_obj.items():
			setattr(config, k, v)
		return config

	@classmethod
	def from_json_file(cls, path: str) -> 'Config':
		"""Load a configuration object from a file.

		Args:
			path (str): Path to the configuration file.
		
		Returns:
			config (Config): The configuration object.
		"""
		with open(path, 'r', encoding='utf-8') as r:
			return cls.from_dict(json.load(r))

	def to_dict(self) -> Dict[str, Any]:
		"""Convert a configuration object to a dictionary.

		Returns:
			output (Dict[str, Any]): The configuration parameters.
		"""
		output = copy.deepcopy(self.__dict__)
		return output

	def save_config(self, path: str):
		"""Save a configuration object to a file.
		
		Args:
			path (str): Path to save the configuration file.
		"""
		if os.path.isdir(path):
			path = os.path.join(path, 'config.json')
		print('Save config to {}'.format(path))
		with open(path, 'w', encoding='utf-8') as w:
			w.write(json.dumps(self.to_dict(), indent=2, sort_keys=True))