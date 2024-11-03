import os
cwd = os.getcwd()
if cwd.endswith("source/predict"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from keys import API_KEYS
from model.model_zs import ZSModel
import jsonlines
import time
from time import sleep
from tqdm import tqdm
import argparse
from dataset.utils import load_data


if __name__ == "__main__":

	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", type=str)
	Parser.add_argument("--split", help="The split of the dataset.", choices=["train", "dev", "test"])
	Parser.add_argument("--few_shots", help="The number of examples in the few-shot prompt.", type=str)
	Parser.add_argument("--subset", help="The subset of the dataset. The MATH dataset has multiple subsets.", type=str)
	Parser.add_argument("--max_test_num", help="The maximum number of examples to test.", type=int, default=200)
	Parser.add_argument("--strategies", help="The strategies to use.", type=str, default="")
	Parser.add_argument("--model_name", help="The name of the model (should have a corresponding config file under `configuration/config_files/dataset_name` called `{model_name}.json`.)")
	Parser.add_argument("--api_key_ids", help="The API keys to use.", default="['key1']")

	args = Parser.parse_args()
	model_name = args.model_name
	dataset_name = args.dataset_name
	split = args.split
	subset = args.subset
	api_key_ids = eval(args.api_key_ids)
	api_keys = [API_KEYS[api_key_id] for api_key_id in api_key_ids]

	config_frn = f"source/configuration/config_files/{dataset_name}/{model_name}.json"
	config = Config.from_json_file(config_frn)
	config.dataset_name = dataset_name
	config.split = split
	config.api_keys = api_keys
	strategies = args.strategies.split(",")
	config.strategies = [f"{args.few_shots}-{x}" for x in strategies]

	lm = model_name.split("_")[0]
	if dataset_name in ["LLC"]:
		config.subset = subset
		config.examples_dir = f"source/prompt/{dataset_name}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{subset}/{model_name}"
	elif dataset_name in ["MATH"]:
		config.subset = subset
		config.examples_dir = f"source/prompt/{dataset_name}/{subset}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{subset}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{subset}/{model_name}"
	else:
		config.examples_dir = f"source/prompt/{dataset_name}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{model_name}"

	os.makedirs(output_dir, exist_ok=True)
	config.output_dir = output_dir
	print("config: ", vars(config))

	# load the dataset
	if dataset_name in ["MATH"]:
		dataset_frn = f"data/{dataset_name}/{subset}/{split}.json"
	else:
		dataset_frn = f"data/{dataset_name}/{split}.json"

	dataset = load_data(dataset_frn)[:args.max_test_num]

	all_predictions = []
	for strategy in config.strategies:
		predictions_folder = os.path.join(output_dir, strategy)
		predictions_file = os.path.join(predictions_folder, "predictions.jsonl")
		predictions = load_data(predictions_file)[:args.max_test_num]
		all_predictions.append(predictions)

	for i, example in enumerate(dataset):
		candidate_solutions = []
		answers = []

		j = 0
		for predictions in all_predictions:
			solution = predictions[i]["completion"]
			answer = predictions[i]["answer"]
			if answer == "[invalid]":
				continue
			j += 1
			candidate_solutions.append(f"Solution {j}:{solution}")
			answers.append(answer)

		example["candidate_solutions"] = "\n\n".join(candidate_solutions)
		example["answers"] = answers
		example["completions"] = candidate_solutions

	def selection(config):
		output_fwn = f"{config.output_dir}/reasoning_selection_{config.strategies}.jsonl"

		# load the model
		model = ZSModel(config)

		# load existing predictions if any
		line_id = 0
		if os.path.isfile(output_fwn):
			with open(output_fwn, "r") as fr:
				reader = jsonlines.Reader(fr)
				for line_id, line in enumerate(reader):
					example_id = line["id"]
		if line_id > 0:
			start_id = line_id + 1
		else:
			start_id = 0

		print(f"Making predictions on dataset {dataset_name} using model {model_name},\nstarting from the {start_id}th example...")

		with open(output_fwn, 'a') as fw:
			writer = jsonlines.Writer(fw, flush=True)
			t0 = time.time()
			for i, example in tqdm(enumerate(dataset), file=sys.stdout):
				if i < start_id:
					continue

				question = example["question"]
				question_id = int(example["id"]) if "id" in example else i
				print(f"\n*****{question_id}*****\n")

				if len(set(example["answers"])) == 1:
					row = {
						"id": question_id,
						"answer": example["answers"][0],
						"completion": example["completions"][0],
					}
				elif len(example["answers"]) == 0:
					row = {
						"id": question_id,
						"answer": "[invalid]",
						"completion": "",
					}
				else:
					try:
						output = model.predict(example)
						answer = output["answer"]
						completion = output["completion"]
					except Exception as e:
						answer, completion = "[error]", str(e)
						print(f"Error at example {i}: {str(e)}", file=sys.stderr)

					row = {
						"id": question_id,
						"answer": answer,
						"completion": completion,
					}

					print("question: ", question)
					print("answer: ", answer)
					print("completion: ", completion)
					print()

				writer.write(row)

			if i % 5 == 0:
				print(f"Finished {i} examples in {time.time() - t0} seconds.", flush=True)

	selection(config)









