# [FFT] UC1 on Alpaca Magpie on Llama Base
CUDA_VISIBLE_DEVICES=0 py visualization/create_embeddings.py --use_case 1
# MODEL_NAME='meta-llama/Llama-3.2-3B'
# EXISTING_DATA_NAME="Magpie-Llama-3.1-Pro-300K-Filtered"
# NEW_DATA_NAME="Magpie-Llama-3.1-Pro-300K-Filtered"
# source /home/ishikaa2/delift/subset_selection/src/utils/dist_utils/run_icl_blocks.sh
# CUDA_VISIBLE_DEVICES=0,1 python3 visualization/load_all_experiments.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME

# [FFT] UC2 on GSM-8k and MixInstruct on Mistral
CUDA_VISIBLE_DEVICES=1 py visualization/create_embeddings.py --use_case 2
# MODEL_NAME='mistralai/Mistral-7B-v0.1'
# EXISTING_DATA_NAME="openai/benchmark_gsm8k"
# NEW_DATA_NAME="llm-blender/mix-instruct"
# source /home/ishikaa2/delift/subset_selection/src/utils/dist_utils/run_icl_blocks.sh
# CUDA_VISIBLE_DEVICES=0,1 python3 visualization/load_all_experiments.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME

# # [FFT] UC3 on Alpaca and MultiAlpaca on Granite
CUDA_VISIBLE_DEVICES=2 py visualization/create_embeddings.py --use_case 3
MODEL_NAME='ibm-granite/granite-3.1-8b-base'
EXISTING_DATA_NAME="tatsu-lab/alpaca"
NEW_DATA_NAME="temp/chatalpaca-20k" # https://github.com/icip-cas/ChatAlpaca/blob/main/data/chatalpaca-20k.json
# source /home/ishikaa2/delift/subset_selection/src/utils/dist_utils/run_icl_blocks.sh
# CUDA_VISIBLE_DEVICES=0,1 python3 visualization/load_all_experiments.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME