from transformers import AutoModelForCausalLM, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, RobertaConfig, RobertaModel, RobertaTokenizer
from datasets import load_dataset
import torch.nn.functional as F
from scipy.stats import entropy
import pyarrow.compute as pc
from tqdm import tqdm
import pyarrow as pa
import numpy as np
import evaluate
import pickle
import torch

# mix-instruct specific dataset processing
def process_dataset(split):
    ds = load_dataset("llm-blender/mix-instruct", split=split)
    ds = ds.to_pandas()
    ds = ds.rename(columns={'output':'completion'})
    ds['prompt'] = ds['instruction'] + ds['input']
    ds['full_example'] = ds['prompt'] + ds['completion']
    return ds[:1000]

# grade school math
def process_dataset_(ds):
    ds = ds.to_pandas()
    ds = ds.rename(columns={'RESPONSE':'completion'})
    ds['prompt'] = ds['INSTRUCTION']
    ds['full_example'] = ds['prompt'] + ds['completion']
    return ds
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Loading dataset...')
training = process_dataset('train')
validation = process_dataset('validation')
n_train = len(training)
n_valid = len(validation)
k = int(0.3 * n_train)

print(f'Loading Llama-8b model on {device}...')
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# tokenizer.pad_token = tokenizer.eos_token

# model_name = "meta-llama/Meta-Llama-3-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# model.config.pad_token_id = model.config.eos_token_id
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
tokenizer.pad_token = tokenizer.eos_token

configuration = RobertaConfig()
model = RobertaModel(configuration).to(device)
print('Done!')

### INFERENCE UTILS

# calculating metrics
def calc_rouge_bleu(pred, ref):
    prediction = [pred]
    reference = [ref]
    final_results = {}

    # BLEU
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=prediction, references=reference)
    final_results['bleu'] = results['bleu']

    # ROUGE
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=prediction, references=reference)
    final_results.update(results)

    return final_results

### Finding k nearest neighbors

# embedding model
print('Loading embedding model BGE...')
embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
embedding_model.eval()

if embedding_tokenizer.pad_token is None:
    embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    embedding_model.resize_token_embeddings(len(embedding_tokenizer))
print('Done!')

# finding knn for query sim
