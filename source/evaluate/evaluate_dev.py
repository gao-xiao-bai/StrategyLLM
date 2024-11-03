import os
if os.getcwd().endswith("evaluate"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data, evaluate_acc
from configuration.configuration import Config
import argparse
import jsonlines
import json


if __name__ == "__main__":
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", type=str)
	Parser.add_argument("--split", help="The split of the dataset.", choices=["train", "dev", "test"])
	Parser.add_argument("--few_shots", help="The number of examples in the few-shot prompt.", type=str)
	Parser.add_argument("--subset", help="The subset of the dataset. The MATH dataset has multiple subsets.", type=str)
	Parser.add_argument("--max_dev_num", help="The maximum number of examples to evaluate.", type=int, default=100)
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
	config.few_shots = few_shots

	if dataset_name in ["MATH"]:
		config.subset = subset
		output_dir = f"output_dir/{dataset_name}/{split}/{subset}/{model_name}/{few_shots}"
	else:
		output_dir = f"output_dir/{dataset_name}/{split}/{model_name}/{few_shots}"

	config.output_dir = output_dir
	config.dataset_name = dataset_name
	print("config: ", vars(config))

	# load the dataset
	if dataset_name in ["MATH"]:
		dataset_frn = f"data/{dataset_name}/{subset}/{split}.json"
	else:
		dataset_frn = f"data/{dataset_name}/{split}.json"

	dataset = load_data(dataset_frn)[:args.max_dev_num]

	def evaluate(config):
		results = []
		for i in range(10):
			pred_frn = f"{config.output_dir}/predictions_{config.few_shots}-{i + 1}.jsonl"
			if not os.path.exists(pred_frn):
				continue

			with open(pred_frn) as fr:
				reader = jsonlines.Reader(fr)
				predictions = [line for line in reader]

			acc, correct_examples, incorrect_examples = evaluate_acc(
				dataset=dataset,
				predictions=predictions,
				dataset_name=dataset_name,
				non_empty_only=non_empty_only,
				valid_only=valid_only,
				debug=debug
			)
			results.append((acc, f"{config.few_shots}-{i + 1}"))

			print(f"Dataset: {dataset_name}\nSplit: {split}\nModel: {model_name}")
			if valid_only:
				print("--valid_only")
			if non_empty_only:
				print("--non_empty_only")
			if debug:
				print("--debug")
			print(f"Answer accuracy: {acc}")

		results.sort(key=lambda x: -x[0])

		with open(f"{config.output_dir}/strategy_validation_results.json", "w") as f:
			json.dump(results, f, indent=4)

	evaluate(config)


