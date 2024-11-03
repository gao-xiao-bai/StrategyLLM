# Configuration

This directory contains code and files related to the model configuration, in order to specify the hyperparameters for each model, such as `"prompt_name"`, `"LM"`, and so on. 

The `configuration.py` file contains the Config class. See the definition of each field in the `init()` funciton.

The `config_files/` directory contains model configuration files on all evaluation datasets. Each dataset has a subdirectory. Under it, each configuration file has the name in the format of `{model_name}.json`, e.g., `gpt-3.5-turbo-16k-0613_cot.json`. 
The fields specify hyperparameter values corresponding to those in `source/configuration/configuration.py`.