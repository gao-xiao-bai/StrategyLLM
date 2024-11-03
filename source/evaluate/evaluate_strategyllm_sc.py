import os
if os.getcwd().endswith("evaluate"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data, is_correct, post_process, extract_gold_answer, extract_pred_answer
from configuration.configuration import Config
import argparse
import jsonlines
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple


def evaluate_acc(
		dataset: list,
		predictions: list, 
		dataset_name: str, 
		non_empty_only: bool = False,
		valid_only: bool = False,
		debug: bool = False) -> Tuple[float, List[Any], List[Any]]:
	"""Evaluate the accuracy of the predictions.

	Args:
		dataset (list): The dataset.
		predictions (list): The predictions.
		dataset_name (str): The name of the dataset.
		non_empty_only (bool): Whether to only consider non-empty predictions.
		valid_only (bool): Whether to only consider valid predictions.
		debug (bool): Whether to only run on the first 10 examples.
	
	Returns:
		Tuple[float, List[Any], List[Any]: The accuracy, the predicted answers, and the gold answers.
	"""
	correct_count, total_count = 0, 0
	gold_answers = []
	pred_answers = []
	for i, (example, prediction) in enumerate(zip(dataset, predictions)):
		gold_id = int(example["id"]) if "id" in example else i
		if prediction == {}:
			continue
		pred_id = int(prediction["id"])

		try:
			assert gold_id == pred_id
		except:
			raise AssertionError(f"Gold id {gold_id} doesn't match pred id {pred_id}.")

		try:
			gold_answer = extract_gold_answer(dataset_name, example["answer"])
		except SyntaxError as e:
			print("Error: ", e)
			print(gold_id)
			exit(-1)
		pred_answer = extract_pred_answer(dataset_name, post_process(prediction["answer"], dataset_name))

		gold_answers.append(gold_answer)
		pred_answers.append(pred_answer)

		if non_empty_only and pred_answer == "":
			continue

		if valid_only:
			if type(pred_answer) == str and ("invalid" in pred_answer or "error" in pred_answer):
				continue

		total_count += 1

		try:
			correct = is_correct(pred_answer, gold_answer)
		except Exception as e:
			exit(-1)

		if correct:
			correct_count += 1
		
		if debug and total_count >= 10:
			break

	acc = round(correct_count / total_count * 100, 1)
	return acc, pred_answers, gold_answers


if __name__ == "__main__":
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", type=str)
	Parser.add_argument("--split", help="The split of the dataset.", choices=["train", "dev", "test"])
	Parser.add_argument("--few_shots", help="The number of examples in the few-shot prompt.", type=str)
	Parser.add_argument("--subset", help="The subset of the dataset. The MATH dataset has multiple subsets.", type=str)
	Parser.add_argument("--max_test_num", help="The maximum number of examples to test.", type=int, default=200)
	Parser.add_argument("--strategies", help="The strategies to use.", type=str, default="")
	Parser.add_argument("--model_name", help="The name of the model (should have a corresponding config file under `configuration/config_files/dataset_name` called `{model_name}.json`.)")
	Parser.add_argument("--non_empty_only", help="If true, only evaluate on non-empty answers.", action="store_true")
	Parser.add_argument("--valid_only", help="If true, only evaluate on valid answers.", action="store_true")
	Parser.add_argument("--debug", help="If true, only run on the first 10 examples.", action="store_true")

	args = Parser.parse_args()
	model_name = args.model_name
	dataset_name = args.dataset_name
	split = args.split
	debug = args.debug
	few_shots = args.few_shots
	subset = args.subset
	non_empty_only = args.non_empty_only
	valid_only = args.valid_only

	config_frn = f"source/configuration/config_files/{dataset_name}/{model_name}.json"
	config = Config.from_json_file(config_frn)
	config.strategies = args.strategies.split(",")
	config.few_shots = few_shots

	if dataset_name in ["MATH"]:
		config.subset = subset
		output_dir = f"output_dir/{dataset_name}/{split}/{subset}/{model_name}"
	else:
		output_dir = f"output_dir/{dataset_name}/{split}/{model_name}"

	config.output_dir = output_dir
	config.dataset_name = dataset_name
	print("config: ", vars(config))

	# load the dataset
	if dataset_name in ["MATH"]:
		dataset_frn = f"data/{dataset_name}/{subset}/{split}.json"
	else:
		dataset_frn = f"data/{dataset_name}/{split}.json"

	dataset = load_data(dataset_frn)[:args.max_test_num]

	def evaluate(config):
		all_answers = []
		gold_answers = []
		for strategy in config.strategies:
			pred_frn = f"{config.output_dir}/{config.few_shots}-{strategy}/predictions.jsonl"
			if not os.path.exists(pred_frn):
				continue

			with open(pred_frn) as fr:
				reader = jsonlines.Reader(fr)
				predictions = [line for line in reader]

			acc, pred_answers, gold_answers = evaluate_acc(
				dataset=dataset,
				predictions=predictions,
				dataset_name=dataset_name,
				non_empty_only=non_empty_only,
				valid_only=valid_only,
				debug=debug
			)
			all_answers.append(pred_answers)

			print(f"Dataset: {dataset_name}\nSplit: {split}\nModel: {model_name}")
			if valid_only:
				print("--valid_only")
			if non_empty_only:
				print("--non_empty_only")
			if debug:
				print("--debug")
			print(f"Answer accuracy: {acc}")

			with open(os.path.join(f"{config.output_dir}/{config.few_shots}-{strategy}", f"result.txt"), "w") as f:
				f.write(f"Dataset: {dataset_name}\nSplit: {split}\nModel: {model_name}\nAnswer accuracy: {acc}")

		return all_answers, gold_answers

	all_answers, gold_answers = evaluate(config)
	new_all_answers = []
	for i in range(len(all_answers[0])):
		answers = []
		for j in range(len(all_answers)):
			answers.append(all_answers[j][i])
		new_all_answers.append(answers)

	correct_count = 0
	coverage = 0
	total_count = 0
	for pred_answers, gold_answer, example in zip(new_all_answers, gold_answers, dataset):
		gold_id = int(example["id"]) if "id" in example else total_count
		total_count += 1
		pred_answers = [x for x in pred_answers if x != "[invalid]" and x != "[error]"]
		if gold_answer in pred_answers:
			coverage += 1

		if not len(pred_answers):
			continue

		counter = Counter(pred_answers)
		pred_answer = counter.most_common(1)[0][0]

		try:
			correct = is_correct(pred_answer, gold_answer)
		except Exception as e:
			exit(-1)

		if correct:
			correct_count += 1
		
	acc = round(correct_count / total_count * 100, 2)
	coverage = round(coverage / total_count * 100, 2)

	print(f"Dataset: {dataset_name}\nSplit: {split}\nModel: {model_name}")
	print(f"total count: {total_count}")
	print(f"Answer accuracy with consistency: {acc}")
	print(f"Coverage: {coverage}")
	with open(os.path.join(config.output_dir, f"result_sc_{args.strategies}.txt"), "w") as f:
		f.write(f"Dataset: {dataset_name}\nSplit: {split}\nModel: {model_name}\nAnswer accuracy: {acc}")


