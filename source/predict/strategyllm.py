import json
import os
cwd = os.getcwd()
if cwd.endswith("source/predict"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from keys import API_KEYS
from model.strategy_master import StrategyMaster
from dataset.utils import load_data, TASK_DEFINITIONS
import argparse


if __name__ == "__main__":

	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", type=str)
	Parser.add_argument("--split", help="The split of the dataset.", choices=["train", "dev", "test"])
	Parser.add_argument("--max_iterations", help="", type=int)
	Parser.add_argument("--few_shots", help="The number of examples in the few-shot prompt.", type=str)
	Parser.add_argument("--subset", help="The subset of the dataset. The MATH dataset has multiple subsets.", type=str)
	Parser.add_argument("--threshold", help="The threshold for strategy execution.", type=float)
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
	config.max_iterations = args.max_iterations
	config.threshold = args.threshold
	config.few_shots = few_shots

	lm = model_name.split("_")[0]
	if dataset_name in ["DU", "WS", "MA","StrategyQA", "LLC"]:
		config.task_definition = TASK_DEFINITIONS[dataset_name]
		config.examples_dir = f"source/prompt/{dataset_name}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{model_name}/{few_shots}"
	elif dataset_name in ["MATH"]:
		config.subset = subset
		config.task_definition = f"The task is to solve all the problems in the {' '.join(subset.split('_'))} subject."
		config.examples_dir = f"source/prompt/{dataset_name}/{subset}/{lm}"
		config.template_dir = f"source/prompt/{dataset_name}/{subset}/{lm}"
		output_dir = f"output_dir/{dataset_name}/{split}/{subset}/{model_name}/{few_shots}"
	else:
		assert False, f"The dataset {dataset_name} is not defined."

	os.makedirs(output_dir, exist_ok=True)
	config.output_dir = output_dir
	print("config: ", vars(config))

	k = max([config.n_votes // 2 + config.n_votes % 2, 10])
	if not os.path.exists(os.path.join(config.examples_dir, f"strategyllm_prompt_{few_shots}-{k}.txt")):
		program_master = StrategyMaster(config)
		print(f"Optimizing programs on dataset {dataset_name} using model {model_name}...")
		final_execution_results = program_master.run()

		with open(os.path.join(config.output_dir, "prompts.json"), "w") as f:
			prompt_dict = {}
			for i, (strategy, execution_results, _) in enumerate(final_execution_results):
				prompt_dict[i] = {
					"strategy": strategy,
					"execution_results": execution_results,
				}
			json.dump(prompt_dict, f, indent=4)

		for i, (strategy, execution_results, _) in enumerate(final_execution_results):
			with open(os.path.join(config.examples_dir, f"strategyllm_prompt_{few_shots}-{i + 1}.txt"), "w") as f:
				execution_results = "\n\n\n\n".join(execution_results)
				f.write(f"Strategy:\n{strategy.strip()}\n\nExamples:\n{execution_results.strip()}")




