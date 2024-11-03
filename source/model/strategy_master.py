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


class StrategyMaster(Base):
	def __init__(self, config: Config):
		super(StrategyMaster, self).__init__(config)

		self.max_iterations = config.max_iterations
		self.threshold = config.threshold
		self.max_num_strategies = 10  # We keep a maximum of 10 strategies

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

	def strategy_generator(self) -> List[str]:
		"""Generate initial strategies to solve the task.

		Returns:
			list: A list of initial strategies.
		"""
		prompt = f"""
Task:
{self.task_definition}
		
Some examples of the task are as follows:
{self.examples}
		
Let's understand the task and write a strategy that consists of a sequence of subtasks to solve the task. For writing, you must satisfy the following requirements:
- Include all necessary subtasks.
- All subtasks are easy to solve.
- Subtasks are in an appropriate order.
- Do not include specific information in the examples.
- Make sure the strategy is general and concise.
The result must be a numbered list in the following format:
1. First subtask
2. Second subtask
"""

		print(f'\n****STRATEGY GENERATOR PROMPT****\n{prompt}\n')
		# query the LM to get the completions
		n = self.n_votes // self.batch_size  # number of iterations to query the LM
		strategies = []
		for iter_id in range(n):
			new_strategies = self._query(prompt=prompt,
										  n=self.batch_size,
										  stop=None,
										  max_tokens=1000,
										  LM=self.LM,
										  temperature=self.temperature)
			strategies += new_strategies

		for i, strategy in enumerate(strategies):
			if "Strategy:" in strategy:
				strategies[i] = strategy.split("Strategy:")[-1].strip()
			elif "Strategy to solve the task:" in strategy:
				strategies[i] = strategy.split("Strategy to solve the task:")[-1].strip()

		with open(os.path.join(self.output_dir, f"initial_strategies.json"), "w") as f:
			json.dump(strategies, f, indent=4)

		for i, strategy in enumerate(strategies):
			print(f'\n****STRATEGY GENERATOR RESPONSE {i+1}****\n{strategy}\n')

		return strategies

	def strategy_executor(self, strategies: List[str], iteration: int) -> Tuple[List[Tuple[str, List[str], float]], List[Tuple[str, List[str], float]]]:
		"""Execute the strategies on the examples.

		Args:
			strategies (list): A list of strategies to be executed.
			iteration (int): The current iteration number.

		Returns:
			tuple: A tuple of correct execution results and incorrect execution results.
		""" 
		qualified_execution_results = []
		unqualified_execution_results = []
		qualified_execution_results_path = os.path.join(self.output_dir, f"qualified_execution_results_{iteration}.json")
		unqualified_execution_results_path = os.path.join(self.output_dir, f"unqualified_execution_results_{iteration}.json")
		if os.path.exists(qualified_execution_results_path) and os.path.exists(unqualified_execution_results_path):
			with open(qualified_execution_results_path) as f:
				qualified_execution_results = json.load(f)
			with open(unqualified_execution_results_path) as f:
				unqualified_execution_results = json.load(f)

		num = len(qualified_execution_results) + len(unqualified_execution_results)
		# print("num: ", num)

		for strategy in strategies[num:]:
			execution_results = []
			execution_accuracy = []
			for example in self.examples_list:
				prompt = f"""
Task:
{self.task_definition}
				
Example of the task:
{example}

Strategy:
{strategy}

The strategy consists of a sequence of subtasks for solving the task. Please execute the strategy on the provided example.
For executing, you need to write a step-by-step solution to the example based on the subtasks. The solution must satisfy the following requirements:
- Adjust and execute these subtasks for this example.
- Compute as many intermediate results as possible. 
- The answer obtained from the solution must be the same as the original answer.
The result must be in the following format:
Question: Question in the provided example
Solution: Solution obtained based on the subtasks in the strategy
Answer: Answer in the provided example, which must include the string 'The answer is '
"""
				print(f'\n****STRATEGY EXECUTOR PROMPT****\n{prompt}\n')
				execution_result = self._query(prompt=prompt,
										  n=1,
										  stop=None,
										  max_tokens=1000,
										  LM=self.LM,
										  temperature=0)[0].strip()
				print(f'\n****STRATEGY EXECUTOR RESPONSE****\n{execution_result}\n')
				predict_answer = extract_answer(execution_result, self.dataset_name)
				gold_answer = extract_answer(example, self.dataset_name)
				if predict_answer == gold_answer:
					print("The execution is correct!")
					execution_accuracy.append(1)
				else:
					print("The execution is incorrect!")
					execution_accuracy.append(0)
				execution_results.append(execution_result)
			execution_accuracy = sum(execution_accuracy) / len(execution_accuracy)

			if execution_accuracy >= self.threshold:
				qualified_execution_results.append((strategy, execution_results, execution_accuracy))
			else:
				unqualified_execution_results.append((strategy, execution_results, execution_accuracy))

			with open(os.path.join(self.output_dir, f"qualified_execution_results_{iteration}.json"), "w") as f:
				json.dump(qualified_execution_results, f, indent=4)

			with open(os.path.join(self.output_dir, f"unqualified_execution_results_{iteration}.json"), "w") as f:
				json.dump(unqualified_execution_results, f, indent=4)

		return qualified_execution_results, unqualified_execution_results

	def strategy_optimizer(self, unqualified_execution_results: List[Tuple[str, List[str], float]], strategies_path: str, feedbacks_path: str) -> List[str]:
		"""Optimize the strategies based on the feedback.

		Args:
			unqualified_execution_results (list): A list of incorrect execution results.
			strategies_path (str): The path to save the optimized strategies.
			feedbacks_path (str): The path to save the feedbacks.
		
		Returns:
			list: A list of optimized strategies.
		"""
		feedbacks = []
		edited_strategies = []
		for strategy, execution_results, _ in unqualified_execution_results:
			feedback = self.examination(strategy, execution_results)
			feedbacks.append([strategy, feedback])
			edited_strategy = self.editing(strategy, feedback)
			edited_strategies.append(edited_strategy)

			with open(feedbacks_path, "w") as f:
				json.dump(feedbacks, f, indent=4)

			with open(strategies_path, "w") as f:
				json.dump(edited_strategies, f, indent=4)

		return edited_strategies

	def examination(self, strategy, execution_results: List[str]) -> str:
		"""Examine the execution results of the strategy.

		Args:
			strategy (str): The strategy to be examined.
			execution_results (list): A list of execution results obtained by executing the strategy on the examples.
		
		Returns:
			str: The feedback obtained by examining the execution results.
		"""
		examination_result = []
		for i, (example, execution_result) in enumerate(zip(self.examples_list, execution_results)):
			predict_answer = extract_answer(execution_result, self.dataset_name)
			gold_answer = extract_answer(example, self.dataset_name)
			suffix = "the same"
			if predict_answer != gold_answer:
				suffix = "different"
			examination_result.append(f"Example {i}:\n{example}\nExecution result obtained by executing the strategy on the example:\n{execution_result}\nThe answer extracted from the execution result is {predict_answer} and the correct answer is {gold_answer}. They are {suffix}.")
		examination_result = "\n\n\n\n".join(examination_result)
		prompt = f"""
Task:
{self.task_definition}

Strategy:
{strategy}

Examination results obtained by executing the strategy on the provided examples of the task and examining the execution results:
{examination_result}

We can see that we do not get the correct answer after executing this strategy on some of the provided examples.
Please carefully analyze why the answers extracted from the execution results of these examples are incorrect and provide suggestions for improving the strategy.
"""
		print(f'\n****EXAMINATION AGENT PROMPT****\n{prompt}\n')
		examination_strategy_result = self._query(prompt=prompt,
								 n=1,
								 stop=None,
								 max_tokens=1000,
								 LM=self.LM,
								 temperature=0)[0].strip()
		print(f'\n****EXAMINATION AGENT RESPONSE****\n{examination_strategy_result}\n')

		return examination_strategy_result

	def editing(self, original_strategy: str, feedback: str) -> str:
		"""Edit the original strategy based on the feedback.

		Args:
			original_strategy (str): The original strategy to be edited.
			feedback (str): The feedback obtained by examining the execution results.

		Returns:
			str: The edited strategy.
		"""
		prompt = f"""
Task:
{self.task_definition}

Some examples of the task are as follows:
{self.examples}

Original strategy to solve the task:
{original_strategy}

Feedback:
{feedback}

You need to revise the original strategy based on the feedback to obtain a better strategy. The newly obtained strategy must be a numbered list in the following format:
1. First subtask
2. Second subtask
"""
		print(f'\n****EDITING AGENT PROMPT****\n{prompt}\n')
		editing_result = self._query(prompt=prompt,
										n=1,
										stop=None,
										max_tokens=1000,
										LM=self.LM,
										temperature=0)[0].strip()
		print(f'\n****EDITING AGENT RESPONSE****\n{editing_result}\n')
		if "Revised strategy to solve the task:" in editing_result:
			editing_result = editing_result.split("Revised strategy to solve the task:")[-1].strip()
		if "Strategy:" in editing_result:
			editing_result = editing_result.split("Strategy:")[-1].strip()
		if "strategy:" in editing_result:
			editing_result = editing_result.split("strategy:")[-1].strip()
		if "Note:" in editing_result:
			editing_result = editing_result.split("Note:")[0].strip()

		return editing_result

	def run(self) -> List[Tuple[str, List[str], float]]:
		"""Run the strategy master.

		Returns:
			list: A list of final execution results.
		"""
		iteration = 0
		initial_strategies_path = os.path.join(self.output_dir, f"initial_strategies.json")
		if os.path.exists(initial_strategies_path):
			with open(initial_strategies_path) as f:
				strategies = json.load(f)
		else:
			strategies = self.strategy_generator()

		final_execution_results = []
		while iteration < self.max_iterations:
			iteration += 1
			execution_results_path = os.path.join(self.output_dir, f"final_execution_results_{iteration}.json")
			unqualified_execution_results_path = os.path.join(self.output_dir, f"unqualified_execution_results_{iteration}.json")

			if os.path.exists(execution_results_path) and os.path.exists(unqualified_execution_results_path):
				with open(execution_results_path) as f:
					final_execution_results = json.load(f)
				with open(unqualified_execution_results_path) as f:
					unqualified_execution_results = json.load(f)
			else:
				qualified_execution_results, unqualified_execution_results = self.strategy_executor(strategies, iteration)
				final_execution_results += qualified_execution_results
				final_execution_results = sorted(final_execution_results, key=lambda x: x[2], reverse=True)
				with open(execution_results_path, "w") as f:
					json.dump(final_execution_results, f, indent=4)

			if len(final_execution_results) >= min([self.n_votes / 2, self.max_num_strategies]):
				break

			if iteration == self.max_iterations:
				break

			strategies_path = os.path.join(self.output_dir, f"strategies_{iteration}.json")
			feedbacks_path = os.path.join(self.output_dir, f"feedbacks_{iteration}.json")
			if os.path.exists(strategies_path):
				with open(strategies_path) as f:
					edited_strategies = json.load(f)
			else:
				edited_strategies = self.strategy_optimizer(unqualified_execution_results, strategies_path, feedbacks_path)

			strategies = edited_strategies

		if len(final_execution_results) < min([self.n_votes / 2, self.max_num_strategies]):
			num = min([self.n_votes // 2 + self.n_votes % 2, self.max_num_strategies]) - len(final_execution_results)
			sorted_unqualified_execution_results = sorted(unqualified_execution_results, key=lambda x: x[2], reverse=True)
			final_execution_results += sorted_unqualified_execution_results[:num]

		return final_execution_results













