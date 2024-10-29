source install.sh

# use case 3: Given a model, its training data, and a new dataset, fine-tune a model on a subset of points from the new dataset that adds new knowledge to the existing dataset
python3 visualization/create_embeddings.py --use_case 3

MODEL_NAME='microsoft/Phi-3-mini-128k-instruct'
python3 visualization/load_all_experiments.py --existing_data_name ibm_ft --new_data_name gov --model_name=$MODEL_NAME
python3 visualization/load_all_experiments.py --existing_data_name squad --new_data_name hotpot_qa --model_name=$MODEL_NAME

MODEL_NAME='Qwen/Qwen2-72B-Instruct'
python3 visualization/load_all_experiments.py --existing_data_name ibm_ft --new_data_name gov --model_name=$MODEL_NAME
python3 visualization/load_all_experiments.py --existing_data_name squad --new_data_name hotpot_qa --model_name=$MODEL_NAME