import os
cwd = os.getcwd()
if cwd.endswith("source/predict"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from keys import API_KEYS
from model.model import Model
from dataset.utils import load_data
import jsonlines
import time
from time import sleep
from tqdm import tqdm
import argparse


if __name__ == "__main__":

	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", type=str)
	Parser.add_argument("--split", help="The split of the dataset.", choices=["train", "dev", "test"])
	Parser.add_argument("--few_shots", help="The number of examples in the few-shot prompt.", type=str)
	Parser.add_argument("--strategy", help="The strategy to use.", type=str)
	Parser.add_argument("--subset", help="The subset of the dataset. The MATH dataset has multiple subsets.", type=str)
	Parser.add_argument("--max_test_num", help="The maximum number of examples to test.", type=int, default=200)
	Parser.add_argument("--model_name", help="The name of the model (should have a corresponding config file under `configuration/config_files/dataset_name` called `{model_name}.json`.)")
	Parser.add_argument("--completion_only", help="Only query the LM to generate the completion (reasoning chain), but not execute the solver to derive the answer.", action="store_true")
	Parser.add_argument("--debug", help="If true, only run on the first 10 examples.", action="store_true")
	Parser.add_argument("--api_key_ids", help="The API keys to use.", default="['key1']")

	args = Parser.parse_args()
	model_name = args.model_name
	dataset_name = args.dataset_name
	split = args.split
	debug = args.debug
	few_shots = f"{args.few_shots}-{args.strategy}"
	subset = args.subset
	completion_only = args.completion_only
	api_key_ids = eval(args.api_key_ids)
	api_keys = [API_KEYS[api_key_id] for api_key_id in api_key_ids]

	config_frn = f"source/configuration/config_files/{dataset_name}/{model_name}.json"
	config = Config.from_json_file(config_frn)
	config.dataset_name = dataset_name
	config.split = split
	config.api_keys = api_keys
	config.few_shots = few_shots

	lm = model_name.split("_")[0]
	if dataset_name in ["LLC"]:
		config.subset = subset
		config.examples_dir = f"source/prompt/{dataset_name}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{subset}/{model_name}/{few_shots}"
	elif dataset_name in ["MATH"]:
		config.subset = subset
		config.examples_dir = f"source/prompt/{dataset_name}/{subset}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{subset}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{subset}/{model_name}/{few_shots}"
	else:
		config.examples_dir = f"source/prompt/{dataset_name}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{model_name}/{few_shots}"

	os.makedirs(output_dir, exist_ok=True)
	config.output_dir = output_dir
	print("config: ", vars(config))

	# load the dataset
	if dataset_name in ["MATH"]:
		dataset_frn = f"data/{dataset_name}/{subset}/{split}.json"
	else:
		dataset_frn = f"data/{dataset_name}/{split}.json"

	dataset = load_data(dataset_frn)[:args.max_test_num]

	def predict(config):
		output_fwn = f"{config.output_dir}/predictions{'_completion_only' if completion_only else ''}{'_debug' if debug else ''}.jsonl"

		# load the model
		model = Model(config)

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
				if debug and i >= 10:
					break
				if i < start_id:
					continue
				question = example["question"]
				question_id = int(example["id"]) if "id" in example else i
				try:
					output = model.predict(example, completion_only=completion_only)
					answer = output["answer"]
					initial_answers = output["initial_answers"]
					answers = output["answers"]
					completion = output["completion"]
					completions = output["completions"]
				except Exception as e:
					answer, initial_answers, answers, completion, completions = "[error]", [], [], str(e), ""
					print(f"Error at example {i}: {str(e)}", file=sys.stderr)

				row = {
					"id": question_id,
					"answer": answer,
					"initial_answers": initial_answers,
					"answers": answers,
					"completion": completion,
					"completions": completions
				}
				writer.write(row)

				print(f"\n*****{question_id}*****\n")
				print("question: ", question)
				print("answer: ", answer)
				print("answers: ", answers)
				print("completion: ", completion)
				print()

			if i % 5 == 0:
				print(f"Finished {i} examples in {time.time() - t0} seconds.", flush=True)

	if os.path.exists(os.path.join(config.examples_dir, f"{config.prompt_name}_prompt_{config.few_shots}.txt")):
		predict(config)









