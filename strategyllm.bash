
model_name=$1
dataset_name=$2
subset=$3
threshold=$4
few_shots=$5

python source/predict/strategyllm.py \
  --model_name $model_name \
  --dataset_name $dataset_name \
  --subset $subset \
  --split dev \
  --max_iterations 3 \
  --threshold $threshold \
  --few_shots $few_shots



# e.g.
#python source/predict/strategyllm.py \
#  --model_name gpt-4-0613_generation \
#  --dataset_name MA \
#  --subset None \
#  --split dev \
#  --max_iterations 3 \
#  --threshold 0.75 \
#  --few_shots 3

