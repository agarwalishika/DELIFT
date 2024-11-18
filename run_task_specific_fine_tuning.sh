# source install.sh

# use case 2: Given a model and a new dataset, fine-tune a model on a subset of points that improves the performance on a benchmark.
# python3 visualization/create_embeddings.py --use_case 2

# MODEL_NAME='microsoft/Phi-3-mini-4k-instruct'
# python3 visualization/load_all_experiments.py --existing_data_name hotpot_qa --new_data_name benchmark_mmlu --model_name=$MODEL_NAME
# python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name benchmark_mt_bench_human_judgments --model_name=$MODEL_NAME

MODEL_NAME='mistralai/Mistral-7B-v0.1'
# python3 visualization/load_all_experiments.py --existing_data_name hotpot_qa --new_data_name benchmark_mmlu --model_name=$MODEL_NAME
CUDA_VISIBLE_DEVICES=1 python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name benchmark_gsm8k --model_name=$MODEL_NAME

notify "experiments done - benchmark"