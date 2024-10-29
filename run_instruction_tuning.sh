source install.sh

# use case 1: Given a dataset, fine-tune a model on a subset of points that improves the performance on the entire dataset.
python3 visualization/create_embeddings.py --use_case 1

MODEL_NAME='microsoft/Phi-3-mini-128k-instruct'
python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME
python3 visualization/load_all_experiments.py --existing_data_name P3 --new_data_name P3 --model_name=$MODEL_NAME

MODEL_NAME='Qwen/Qwen2-72B-Instruct'
python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME
python3 visualization/load_all_experiments.py --existing_data_name P3 --new_data_name P3 --model_name=$MODEL_NAME
