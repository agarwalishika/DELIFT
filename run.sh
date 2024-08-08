# pip installs
# pip3 install streamlit
# pip3 install scikit-learn
# pip3 install plotly
# export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# pip3 install sklearn
# pip3 install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
# pip3 install sentence-transformers
# pip3 install faiss-gpu
# pip3 install peft
# pip3 install evaluate
# pip3 install torch
# pip3 install transformers
# pip3 install trl
# pip3 install bert-score

# running experiments
# python3 visualization/create_embeddings.py
conda activate env3.10
python3 visualization/load_all_experiments.py
# python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct