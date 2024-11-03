import json
import os
if os.getcwd().endswith("evaluate"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data, evaluate_acc
from configuration.configuration import Config
import argparse
import jsonlines


if __name__ == "__main__":
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", type=str)
	Parser.add_argument("--split", help="The split of the dataset.", choices=["train", "dev", "test"])
	Parser.add_argument("--few_shots", help="The number of examples in the few-shot prompt.", type=str)
	Parser.add_argument("--strategy", help="The strategy to use.", type=str)
	Parser.add_argument("--subset", help="The subset of the dataset. The MATH dataset has multiple subsets.", type=str)
	Parser.add_argument("--max_test_num", help="The maximum number of examples to test.", type=int, default=200)
	Parser.add_argument("--model_name", help="The name of the model (should have a corresponding config file under `configuration/config_files/dataset_name` called `{model_name}.json`.)")
	Parser.add_argument("--non_empty_only", help="If true, only evaluate on non-empty answers.", action="store_true")
	Parser.add_argument("--valid_only", help="If true, only evaluate on valid answers.", action="store_true")
	Parser.add_argument("--debug", help="If true, only run on the first 10 examples.", action="store_true")

	args = Parser.parse_args()
	model_name = args.model_name
	dataset_name = args.dataset_name
	split = args.split
	debug = args.debug
	few_shots = f"{args.few_shots}-{args.strategy}"
	subset = args.subset
	non_empty_only = args.non_empty_only
	valid_only = args.valid_only

	config_frn = f"source/configuration/config_files/{dataset_name}/{model_name}.json"
	config = Config.from_json_file(config_frn)
	config.few_shots = few_shots

	if dataset_name in ["MATH", "LLC"]:
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

	dataset = load_data(dataset_frn)[:args.max_test_num]

	def evaluate(config):

		pred_frn = f"{config.output_dir}/predictions.jsonl"

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

		with open(f"{config.output_dir}/correct_examples.json", "w") as f:
			json.dump(correct_examples, f, indent=4)

		with open(f"{config.output_dir}/incorrect_examples.json", "w") as f:
			json.dump(incorrect_examples, f, indent=4)

		print(f"Dataset: {dataset_name}\nSplit: {split}\nModel: {model_name}")
		if valid_only:
			print("--valid_only")
		if non_empty_only:
			print("--non_empty_only")
		if debug:
			print("--debug")
		print(f"Answer accuracy: {acc}")

		with open(os.path.join(config.output_dir, f"result.txt"), "w") as f:
			f.write(f"Dataset: {dataset_name}\nSplit: {split}\nModel: {model_name}\nAnswer accuracy: {acc}")

	evaluate(config)


