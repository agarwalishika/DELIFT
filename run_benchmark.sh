# pip installs
# pip install streamlit
# pip install scikit-learn
# pip install plotly
# export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# pip install sklearn
# pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
# pip install sentence-transformers
# pip install faiss-gpu
# pip install peft
# pip install evaluate
# pip install torch
# pip install transformers
# pip install trl
# pip install bert-score
# pip install numpy

### RUN EXPERIMENTS FOR OUR METHODOLOGY ###

# use case 2: Given a model and a new dataset, fine-tune a model on a subset of points that improves the performance on a benchmark.
python3 visualization/create_embeddings.py --use_case 2

for i in $(seq 1 10);
do
    echo $i

    MODEL_NAME='microsoft/Phi-3-mini-4k-instruct'
    python3 visualization/load_all_experiments.py --existing_data_name hotpot_qa --new_data_name benchmark_mmlu --model_name=$MODEL_NAME
    python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name benchmark_mt_bench_human_judgments --model_name=$MODEL_NAME

    MODEL_NAME='Qwen/Qwen2-72B-Instruct'
    python3 visualization/load_all_experiments.py --existing_data_name hotpot_qa --new_data_name benchmark_mmlu --model_name=$MODEL_NAME
    python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name benchmark_mt_bench_human_judgments --model_name=$MODEL_NAME

done

notify "experiments done - bench"