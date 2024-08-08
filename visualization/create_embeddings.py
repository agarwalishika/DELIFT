import sys
sys.path.append('.')
import subset_selection.model_inference as mi
from folder_names import FolderNames
from datasets import load_dataset
from sklearn.manifold import TSNE
from models import Models
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import pickle
import torch
import json
import os

model = Models()

# processing functions
mmlu_input_processing = lambda ds: ds.apply(lambda x: x['question'] + ":\n" + "\n".join(["A. ", "B. ", "C. ", "D. "] + x['choices']), axis=1)

# Load the embeddings
def load_matrix(datafile_path):
    with open(datafile_path, 'rb') as f:
        df = pickle.load(f)

    matrix, input = [], []
    for i in range(3): #train, valid, test
        matrix.append(np.array(df[i]['vis_dims'].to_list()).squeeze())
        input.append(np.array(df[i]['data'].to_list()).squeeze())
    return matrix, input

def get_matrix(dataset_pkl_folder):
    """
    Combines all the grouped data into one set of matrices

    Args:
        dataset_pkl_folder: path where the dataset information is stored
    Returns:
        matrices: t-SNE embeddings set
        data: original text data set
    """
    matrices = []
    data = []

    for i, dirname in enumerate(os.listdir(dataset_pkl_folder)):
        ma, d = load_matrix(os.path.join(dataset_pkl_folder, dirname))
        matrices.append(ma)
        data.append(d)

    return matrices, data

# Create a t-SNE model and transform the data
def fit_tsne(matrix):
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    return vis_dims

def parse_hf_datasets(json_file='visualization/huggingface_datasets.json'):
    """
    Parses HuggingFace datasets to be usable for subset selection.

    Args:
        json_file: path to JSON file that contains configurations for HuggingFace data loading
    Returns:
        A dictionary of datasets where:
        - the key is the name of the dataset
        - the value is a tuple of the train, valid, and test splits of the dataset.

        Each split is a Pandas DataFrame with columns: instruction, input, output, and data (which is a combination of the first 3 columns)
    """
    dataset_config = json.load(open(json_file, 'r'))

    full_dataset = {}
    for key in tqdm(dataset_config.keys()):
        # load the dataset
        assert "split_names" in dataset_config[key]

        def get_split_ds(split_name, subset=None):
            if subset:
                ds = load_dataset(key, subset, split=split_name)
            else:
                ds = load_dataset(key, split=split_name)

            # convert to pandas dataset
            ds = ds.to_pandas()

            # rename/create instruction, input, and output columns
            def process_column(ds, col_name, processing_function=None):
                if not col_name in dataset_config[key]:
                    # do something, create it
                    ds[col_name] = ""
                elif "|" in dataset_config[key][col_name]:
                    ds[col_name] = processing_function(ds)
                else:
                    ds = ds.rename(columns={dataset_config[key][col_name]:col_name})
                return ds
            
            ds = process_column(ds, 'instruction')
            ds = process_column(ds, 'input')
            ds = process_column(ds, 'output')

            ds['data'] = "Instruction: " + ds['instruction'] + "\nInput: " + ds['input'] + "\nOutput: " + ds['output']
            return ds.sample(frac=1.0)

        subset = dataset_config[key]["subset"] if "subset" in dataset_config[key].keys() else None
        split_keys = dataset_config[key]['split_names'].split("|")
        train_ds = get_split_ds(split_keys[0], subset)
        valid_ds = get_split_ds(split_keys[1], subset)
        test_ds = get_split_ds(split_keys[2], subset)

        new_key = key[key.rfind('/')+1:]
        full_dataset[new_key] = (train_ds, valid_ds, test_ds)
    
    return full_dataset

def parse_qr_datasets():
    """
    Parses Query Rewriting datasets (IBM and Government) to be usable for subset selection.

    Args:
        None, file paths are assumed to be saved in a FolderNames instance (folder_names.py).
    Returns:
        A dictionary of datasets where:
        - the key is the name of the dataset
        - the value is a tuple of the train, valid, and test splits of the dataset.

        Each split is a Pandas DataFrame with columns: instruction, input, output, and data (which is a combination of the first 3 columns)
    """
    full_dataset = {}

    gov_data = json.load(open(FolderNames.qr_gov_data_file, 'r'))
    og_data = json.load(open(FolderNames.qr_ibm_ft_data_file, 'r'))

    for config in [(gov_data, "gov"), (og_data, "ibm_ft")]:
        dataset, key = config
        data = []
        for i in range(len(dataset)):
            # instruction = "Given the following conversation, please reword the final utterance from the user into an single utterance that doesn't need the history to understand the user's intent. If the final utterance is a clear and standalone question, just RETURN THE FINAL UTTERANCE."
            instruction = "Given the following conversation, rewrite the user's final question or statement as a standalone query, removing any dependency on previous conversation context. If the final utterance is already a clear, self-contained question, simply repeat it verbatim."
            conversation = ""
            for turn in dataset[i]['input']:
                conversation += turn['speaker'] + ": " + turn['text'] + "\n"
            output = f"user: {dataset[i]['query_and_question_info'][1]['query']}"

            # non-standalone
            input = conversation + f"user: {dataset[i]['query_and_question_info'][0]['query']}"
            data.append(f"Instruction: {instruction}\nInput:\n{input}\nOutput:\n{output}\n")

            # standalone
            input = conversation + f"user: {dataset[i]['query_and_question_info'][1]['query']}"
            data.append(f"Instruction: {instruction}\nInput:\n{input}\nOutput:\n{output}\n")

        # split into training/validation/testing sets
        x = len(data)
        train_ds = pd.DataFrame(data[:int(0.7*x)], columns=['data'])
        valid_ds = pd.DataFrame(data[int(0.7*x):int(0.9*x)], columns=['data'])
        test_ds = pd.DataFrame(data[int(0.9*x):], columns=['data'])

        new_key = key[key.rfind('/')+1:]
        full_dataset[new_key] = (train_ds, valid_ds, test_ds)
    return full_dataset

def parse_benchmark_datasets(json_file='visualization/benchmark_datasets.json'):
    """
    Parses HuggingFace benchmark datasets to be usable for subset selection.

    Args:
        json_file: path to JSON file that contains configurations for HuggingFace benchmark data loading
    Returns:
        A dictionary of datasets where:
        - the key is the name of the dataset
        - the value is a tuple of the train, valid, and test splits of the dataset.

        Each split is a Pandas DataFrame with columns: instruction, input, output, and data (which is a combination of the first 3 columns)
    """
    dataset_config = json.load(open(json_file, 'r'))

    full_dataset = {}
    for key in tqdm(dataset_config.keys()):
        # load the dataset
        assert "split_names" in dataset_config[key]

        def get_split_ds(split_name, subset=None):
            if subset:
                ds = load_dataset(key, subset, split=split_name)
            else:
                ds = load_dataset(key, split=split_name)

            # convert to pandas dataset
            ds = ds.to_pandas()

            # rename/create instruction, input, and output columns
            def process_column(ds, col_name, processing_function=None):
                if not col_name in dataset_config[key]:
                    # do something, create it
                    ds[col_name] = ""
                elif "|" in dataset_config[key][col_name]:
                    ds[col_name] = processing_function(ds)
                else:
                    ds = ds.rename(columns={dataset_config[key][col_name]:col_name})
                return ds
            
            ds = process_column(ds, 'instruction')
            if "|" in dataset_config[key]['input'] and "mmlu" in key:
                ds = process_column(ds, 'input', mmlu_input_processing)
            else:
                ds = process_column(ds, 'input')
            ds = process_column(ds, 'output')

            ds['data'] = "Instruction: " + str(ds['instruction']) + "\nInput: " + str(ds['input']) + "\nOutput: " + str(ds['output'])
            return ds.sample(frac=1.0)

        subset = dataset_config[key]["subset"] if "subset" in dataset_config[key].keys() else None
        split_keys = dataset_config[key]['split_names'].split("|")
        train_ds = get_split_ds(split_keys[0], subset)
        valid_ds = get_split_ds(split_keys[1], subset)
        test_ds = get_split_ds(split_keys[2], subset)

        # create new key with "benchmark" tag
        new_key = "benchmark_" + key[key.rfind('/')+1:]
        full_dataset[new_key] = (train_ds, valid_ds, test_ds)
    
    return full_dataset

def extract_prompt(x):
    x = str(x)
    s = 'Output:'
    ind = x.index(s)
    return x[:ind]

def find_embedding(ds):
    """
    Computes embeddings that will be used to fit t-SNE embeddings.

    Args:
        ds: list of data points (combination of instruction, input, and output)
    Returns:
        list of vector embeddings
    """
    l = [extract_prompt(d) for d in list(ds)]
    embedding = mi.batch_inference(model.embedding_model, model.embedding_tokenizer, l, model.device)
    return embedding.cpu()

def main(args):
    print('processing dataset')
    if args.use_case == 1:
        spec_fn = FolderNames(model.model_name, "same_data_cache")
    elif args.use_case == 2:
        spec_fn = FolderNames(model.model_name, "benchmark_cache")
    elif args.use_case == 3:
        spec_fn = FolderNames(model.model_name, "version_cache")

    if os.path.exists(spec_fn.visualization_cache_file):
        return
    
    if args.use_case == 1:
        datasets = parse_hf_datasets()
    elif args.use_case == 2:
        datasets = parse_benchmark_datasets()
        hf_dataset = parse_hf_datasets()
        datasets.update(hf_dataset)
    elif args.use_case == 3:
        datasets = parse_qr_datasets()
    
    print('finding embeddings')
    for key in tqdm(datasets.keys()):
        train_ds = find_embedding(datasets[key][0]['data'])
        valid_ds = find_embedding(datasets[key][1]['data'])
        test_ds = find_embedding(datasets[key][2]['data'])

        embeddings = fit_tsne(torch.concat((train_ds, valid_ds, test_ds)))

        t = len(train_ds)
        v = len(valid_ds)
        datasets[key][0]['vis_dims'] = [embeddings[:t][x] for x in range(len(datasets[key][0]))]
        datasets[key][1]['vis_dims'] =  [embeddings[t:t+v][x] for x in range(len(datasets[key][1]))]
        datasets[key][2]['vis_dims'] = [embeddings[t+v:][x] for x in range(len(datasets[key][2]))]

        pkl_name = key + ".pkl"
        with open(os.path.join(spec_fn.dataset_pkl_folder, pkl_name), 'wb+') as f:
            pickle.dump(datasets[key], f)

    del datasets
    vis_dims, data = get_matrix(spec_fn.dataset_pkl_folder)

    with open(spec_fn.visualization_cache_file, 'wb+') as f:
        pickle.dump((vis_dims, data), f)
    print('dumped in', spec_fn.visualization_cache_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_case", type=int, default=2)
    args = parser.parse_args()
    main(args)