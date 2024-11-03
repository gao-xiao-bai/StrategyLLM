
model_name=$1
dataset_name=$2
subset=$3
few_shots=$4

python source/evaluate/evaluate_dev.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --subset ${subset} \
  --split dev \
  --few_shots ${few_shots}


# e.g.
#python source/evaluate/evaluate_dev.py \
#  --model_name gpt-4-0613_strategyllm \
#  --dataset_name MA \
#  --subset None \
#  --split dev \
#  --few_shots 3