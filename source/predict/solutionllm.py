import json
import os
cwd = os.getcwd()
if cwd.endswith("source/predict"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from keys import API_KEYS
from model.solution_master import SolutionMaster
from dataset.utils import load_data, TASK_DEFINITIONS
import argparse


if __name__ == "__main__":

	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", type=str)
	Parser.add_argument("--split", help="The split of the dataset.", choices=["train", "dev", "test"])
	Parser.add_argument("--few_shots", help="The number of examples in the few-shot prompt.", type=str)
	Parser.add_argument("--subset", help="The subset of the dataset. The MATH dataset has multiple subsets.", type=str)
	Parser.add_argument("--model_name", help="The name of the model (should have a corresponding config file under `configuration/config_files/dataset_name` called `{model_name}.json`.)")
	Parser.add_argument("--api_key_ids", help="The API keys to use.", default="['key1']")

	args = Parser.parse_args()
	model_name = args.model_name
	dataset_name = args.dataset_name
	split = args.split
	few_shots = args.few_shots
	subset = args.subset
	api_key_ids = eval(args.api_key_ids)
	api_keys = [API_KEYS[api_key_id] for api_key_id in api_key_ids]

	config_frn = f"source/configuration/config_files/{dataset_name}/{model_name}.json"
	config = Config.from_json_file(config_frn)
	config.dataset_name = dataset_name
	config.split = split
	config.api_keys = api_keys

	lm = model_name.split("_")[0]
	if dataset_name in ["DU", "WS", "MA", "StrategyQA", "LLC"]:
		config.few_shots = few_shots
		config.task_definition = TASK_DEFINITIONS[dataset_name]
		config.examples_dir = f"source/prompt/{dataset_name}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{model_name}/{few_shots}"
	elif dataset_name in ["MATH"]:
		config.subset = subset
		config.few_shots = few_shots
		config.task_definition = f"The task is to solve all the problems in the {' '.join(subset.split('_'))} subject."
		config.examples_dir = f"source/prompt/{dataset_name}/{subset}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{subset}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{subset}/{model_name}/{few_shots}"
	else:
		assert False, f"The dataset {dataset_name} is not defined."

	output_dir = os.path.join(output_dir, "solutionllm")
	os.makedirs(output_dir, exist_ok=True)
	config.output_dir = output_dir
	print("config: ", vars(config))

	if not os.path.exists(os.path.join(config.examples_dir, f"solutionllm_prompt_{few_shots}.txt")):
		solution_master = SolutionMaster(config)
		results = solution_master.run()

		with open(os.path.join(config.output_dir, "prompts.json"), "w") as f:
			json.dump(results, f, indent=4)

		with open(os.path.join(config.examples_dir, f"solutionllm_prompt_{few_shots}.txt"), "w") as f:
			execution_results = "\n\n\n\n".join(results)
			f.write(execution_results.strip())




