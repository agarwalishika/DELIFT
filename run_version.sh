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

# use case 3: Given a model, its training data, and a new dataset, fine-tune a model on a subset of points from the new dataset that adds new knowledge to the existing dataset
python3 visualization/create_embeddings.py --use_case 3

MODEL_NAME='microsoft/Phi-3-mini-4k-instruct'
python3 visualization/load_all_experiments.py --existing_data_name ibm_ft --new_data_name gov --model_name=$MODEL_NAME

MODEL_NAME='Qwen/Qwen2-7B-Instruct'
python3 visualization/load_all_experiments.py --existing_data_name ibm_ft --new_data_name gov --model_name=$MODEL_NAME

notify "experiments done - version"