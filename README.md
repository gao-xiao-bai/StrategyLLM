# StrategyLLM: Large Language Models for Strategy Generation and Problem Solving

Code for [StrategyLLM](https://arxiv.org/abs/2311.08803) (NeurIPS 2024).

## Requirements
Ensure you have the following Python packages installed:
```
openai == 0.27.8
together >= 1.1.4
replicate >= 0.25.1
```

## Repository Structure
- **`data/`**: Contains evaluation datasets.
- **`output_dir/`**: Model predictions will be saved here.
- **`source/`**: Source code for the framework.
  - **`dataset/utils.py`**: Utility functions.
  - **`configuration/`**: Model configurations, specifying hyperparameters such as `"prompt_name"` and `"LM"`.
    - `configuration.py`: Contains the `Config` class. Refer to the `__init__` function for field definitions.
    - `config_files/`: JSON configuration files for each model, named as `{model_name}.json` (e.g., `gpt-4-0613_cot.json`). Check `configuration/README.md` for more details.
  - **`prompt/`**: Prompt files organized by dataset, with each dataset containing subfolders for different LMs. Each prompt includes:
    - `{prompt_name}_prompt_{few_shots}.txt`: Prompt with few-shot examples.
    - `{prompt_name}_template.txt`: Template for input transformation.
    - These files are used together to generate the final prompt for querying the LLM (see `source/model/model.py` for implementation).
  - **`model/`**:
    - `base.py`: Base class for all models.
    - `strategy_master.py`: Class for executing the StrategyLLM framework.
    - `solution_master.py`: Class for executing the SolutionLLM framework.
    - `model.py`: Class for inference.
    - `model_zs.py`: Class for StrategyLLM-ZS.
  - **`predict/`**:
    - `strategyllm.py`: Execute the StrategyLLM framework to obtain top-k strategy-based few-shot prompts.
    - `solutionllm.py`: Execute the SolutionLLM framework for few-shot prompts.
    - `predict.py`: Generate predictions on a test set; outputs saved to `output_dir/`.
    - `predict_dev.py`: Obtain predictions of top-k strategies on a development set; outputs saved to `output_dir/`.
    - `predict_strategyllm_zs.py`: Generate predictions using StrategyLLM-ZS with top-m strategies (default: m=3).
  - **`evaluate/`**:
    - `evaluate.py`: Evaluate the model on a test set.
    - `evaluate_dev.py`: Evaluate top-k strategies on a development set.
    - `evaluate_strategyllm_sc.py`: Evaluate StrategyLLM-SC on a test set.
    - `evaluate_strategyllm_zs.py`: Evaluate StrategyLLM-ZS on a test set.

## Usage
1. **API Keys**: Provide your OpenAI API keys in `source/keys.py`:
   ```python
   API_KEYS = {
       "key1_nickname": "key1",
       "key2_nickname": "key2",
       ...
   }
   ```

2. **Set Additional API Keys**: Configure your API keys for [together.ai](https://docs.together.ai/docs/quickstart) or [replicate](https://replicate.com/docs/get-started/python).

3. **Run StrategyLLM**: Use `strategyllm.bash` to obtain top-k strategies for a specific task.

4. **Development Set Predictions**: Generate predictions for top-k strategies on a development set using `predict_dev.bash`.

5. **Evaluate Strategies**: Use `evaluate_dev.bash` to evaluate top-k strategies and obtain top-m strategies for testing.

6. **Test Set Predictions**: Execute `predict.bash` to get predictions of a strategy on a test set.

7. **Evaluate a Strategy**: Use `evaluate.bash` for strategy evaluation.

8. **StrategyLLM-ZS Predictions**: Run `predict_strategyllm_zs.bash` to get predictions using StrategyLLM-ZS. We need to specify which strategies to use.

9. **Evaluate StrategyLLM-SC or StrategyLLM-ZS**: Evaluate using `evaluate_strategyllm_sc.bash` or `evaluate_strategyllm_zs.bash`.

## Citation
If you find this repository useful, please star it and cite our paper:
```bibtex
@article{gao2023strategyllm,
  title={StrategyLLM: Large Language Models as Strategy Generators, Executors, Optimizers, and Evaluators for Problem Solving},
  author={Gao, Chang and Jiang, Haiyun and Cai, Deng and Shi, Shuming and Lam, Wai},
  journal={arXiv preprint arXiv:2311.08803},
  year={2023}
}
```