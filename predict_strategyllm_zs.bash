
model_name=$1
dataset_name=$2
subset=$3
few_shots=$4
strategies=$5

python source/predict/predict_strategyllm_zs.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --subset ${subset} \
  --split test \
  --few_shots ${few_shots} \
  --strategies ${strategies} \


# e.g.
#python source/predict/predict_strategyllm_zs.py \
#  --model_name gpt-4-0613_strategyllm \
#  --dataset_name MA \
#  --subset None \
#  --split test \
#  --few_shots 3 \
#  --strategies 1,2,3