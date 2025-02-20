#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 /home/ishikaa2/delift/subset_selection/src/utils/dist_utils/get_icl_utility_kernel.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME & \
CUDA_VISIBLE_DEVICES=1 python3 /home/ishikaa2/delift/subset_selection/src/utils/dist_utils/get_icl_utility_kernel.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME --is_data="False" & \
sleep 120s ; \
CUDA_VISIBLE_DEVICES=2 python3 /home/ishikaa2/delift/subset_selection/src/utils/dist_utils/get_icl_utility_kernel.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME & \
CUDA_VISIBLE_DEVICES=3 python3 /home/ishikaa2/delift/subset_selection/src/utils/dist_utils/get_icl_utility_kernel.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME --is_data="False"
notify "OD LORDIE HELP"