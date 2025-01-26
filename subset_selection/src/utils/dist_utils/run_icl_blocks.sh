#!/bin/bash

# Initialize exit_code to a value that doesn't stop the loop
exit_code=0

# Keep running the Python script until the exit code is 14
while [ $exit_code -ne 14 ]; do
    CUDA_VISIBLE_DEVICES=0 python3 get_icl_utility_kernel.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME & \
    CUDA_VISIBLE_DEVICES=1 python3 get_icl_utility_kernel.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME --is_data="False" & \
    sleep 20s & \
    CUDA_VISIBLE_DEVICES=2 python3 get_icl_utility_kernel.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME & \
    CUDA_VISIBLE_DEVICES=3 python3 get_icl_utility_kernel.py --existing_data_name=$EXISTING_DATA_NAME --new_data_name=$NEW_DATA_NAME --model_name=$MODEL_NAME --is_data="False"
    exit_code=$?
    echo "Exit code: $exit_code"
done

echo "Python script exited with code 14. Stopping."
