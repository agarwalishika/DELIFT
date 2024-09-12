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

            ds['data'] = "Instruction: " + ds['instruction'].astype(str) + "\nInput: " + ds['input'].astype(str) + "\nOutput: " + ds['output'].astype(str)
            return ds.sample(n=30)

        subset = dataset_config[key]["subset"] if "subset" in dataset_config[key].keys() else None
        split_keys = dataset_config[key]['split_names'].split("|")
        if "natural" in key:
            temp_ds = get_split_ds(split_keys[0], subset)
            natural_valid = int(0.7 * len(train_ds))
            valid_ds = temp_ds[natural_valid:]
            train_ds = temp_ds[:natural_valid]
            test_ds = get_split_ds(split_keys[1], subset)
        else:
            train_ds = get_split_ds(split_keys[0], subset)
            valid_ds = get_split_ds(split_keys[1], subset)
            test_ds = get_split_ds(split_keys[2], subset)

        new_key = key[key.rfind('/')+1:]
        full_dataset[new_key] = (train_ds, valid_ds, test_ds)
    
    return full_dataset

def parse_qa_datasets():
    """
    Parses Question Answering datasets (SQuAD and HotpotQA) to be usable for subset selection.

    Args:
        None, file paths are assumed to be saved in a FolderNames instance (folder_names.py).
    Returns:
        A dictionary of datasets where:
        - the key is the name of the dataset
        - the value is a tuple of the train, valid, and test splits of the dataset.

        Each split is a Pandas DataFrame with columns: instruction, input, output, and data (which is a combination of the first 3 columns)
    """
    full_dataset = {}

    ds = load_dataset('hotpotqa/hotpot_qa', 'fullwiki')
    data = []
    for dset in [ds['train'], ds['validation'], ds['test']]:
        for i in dset:
            instruction = i['question']
            output = i['answer']

            input = ""
            temp = i['supporting_facts']
            for x in range(len(temp['title'])):
                ind = i['context']['title'].index(temp['title'][x])
                input += i['context']['sentences'][ind][temp['sent_id'][x]] + " "

            data.append(f"Instruction:\nContext: {instruction}\nInput:\n{input}\nOutput:\n{output}\n")

            if len(data) > 10:
                break
        break

    train_ds = pd.DataFrame(data[:int(0.7*x)], columns=['data'])
    valid_ds = pd.DataFrame(data[int(0.7*x):int(0.9*x)], columns=['data'])
    test_ds = pd.DataFrame(data[int(0.9*x):], columns=['data'])
    full_dataset['hotpot_qa'] = (train_ds, valid_ds, test_ds)

    data = []
    ds = load_dataset("rajpurkar/squad")
    for dset in [ds['train'], ds['validation']]:
        for i in dset:
            instruction = i['question']
            input = i['context'][i['answers']['answer_start'][0]:]
            output = i['answers']['text'][0]

            data.append(f"Instruction:\nContext: {instruction}\nInput:\n{input}\nOutput:\n{output}\n")
        
            if len(data) > 10:
                break
        break
    
    train_ds = pd.DataFrame(data[:int(0.7*x)], columns=['data'])
    valid_ds = pd.DataFrame(data[int(0.7*x):int(0.9*x)], columns=['data'])
    test_ds = pd.DataFrame(data[int(0.9*x):], columns=['data'])
    full_dataset['squad'] = (train_ds, valid_ds, test_ds)

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

def mt_bench_processing(d):
    if 'a' in d['winner']:
        conv_key = 'conversation_a'
    else:
        conv_key = 'conversation_b'
    conversation = ""
    for i in range(3):
        conversation += d[conv_key][i]['role'] + ": " + d[conv_key][i]['content'].replace('\n', ' ') + "\n"
    output = f"assistant: {d[conv_key][3]['content']}".replace('\n', ' ')
    return "", conversation, output, "Instruction: " + str("") + "\nInput: " + str(conversation) + "\nOutput: " + str(output)

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

            ds['data'] = "Instruction: " + ds['instruction'].astype(str) + "\nInput: " + ds['input'].astype(str) + "\nOutput: " + ds['output'].astype(str)
            return ds.sample(frac=1.0)

        if 'special_processing' in dataset_config[key]:
            ds = load_dataset(key, split=dataset_config[key]['split_names'])
            df = {'instruction': [], 'input': [], 'output': [], 'data': []}

            if 'mt_bench' in key:
                for d in ds:
                    instruction, input, output, data = mt_bench_processing(d)
                    df['instruction'].append(instruction)
                    df['input'].append(input)
                    df['output'].append(output)
                    df['data'].append(data)
                df = pd.DataFrame.from_dict(df)             
            
            x = len(df)
            train_ds = pd.DataFrame(df[:int(0.7*x)], columns=['data'])
            valid_ds = pd.DataFrame(df[int(0.7*x):int(0.9*x)], columns=['data'])
            test_ds = pd.DataFrame(df[int(0.9*x):], columns=['data'])
        else:
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
    embedding = mi.batch_inference(model.embedding_model, model.embedding_tokenizer, l)
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
        qa_dataset = parse_qa_datasets()
        datasets.update(qa_dataset)
    elif args.use_case == 3:
        datasets = parse_qr_datasets()
        qa_datasets = parse_qa_datasets()
        datasets.update(qa_datasets)
    
    print('finding embeddings')
    for key in tqdm(datasets.keys()):
        pkl_name = key + ".pkl"
        print(os.path.exists(os.path.join(spec_fn.dataset_pkl_folder, pkl_name)))
        if not os.path.exists(os.path.join(spec_fn.dataset_pkl_folder, pkl_name)):
            train_ds = find_embedding(datasets[key][0]['data'])
            valid_ds = find_embedding(datasets[key][1]['data'])
            test_ds = find_embedding(datasets[key][2]['data'])

            embeddings = fit_tsne(torch.concat((train_ds, valid_ds, test_ds)))

            t = len(train_ds)
            v = len(valid_ds)
            datasets[key][0]['vis_dims'] = [embeddings[:t][x] for x in range(len(datasets[key][0]))]
            datasets[key][1]['vis_dims'] =  [embeddings[t:t+v][x] for x in range(len(datasets[key][1]))]
            datasets[key][2]['vis_dims'] = [embeddings[t+v:][x] for x in range(len(datasets[key][2]))]

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