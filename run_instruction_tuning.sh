# source install.sh

# use case 1: Given a dataset, fine-tune a model on a subset of points that improves the performance on the entire dataset.
# python3 visualization/create_embeddings.py --use_case 1

# MODEL_NAME='microsoft/Phi-3-mini-128k-instruct'
# python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME
# python3 visualization/load_all_experiments.py --existing_data_name P3 --new_data_name P3 --model_name=$MODEL_NAME

# MODEL_NAME='facebook/opt-125m'
# CUDA_VISIBLE_DEVICES=0 python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME
MODEL_NAME='meta-llama/Llama-3.2-3B'
CUDA_VISIBLE_DEVICES=0 python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME
# python3 visualization/load_all_experiments.py --existing_data_name P3 --new_data_name P3 --model_name=$MODEL_NAME

notify "experiments done - instruction tuning"