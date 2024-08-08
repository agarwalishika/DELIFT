#imports
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import torch
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class InferenceICL():
    def __init__(self, model, tokenizer) -> None:
        self.embedding_model = model.to(device)
        self.embedding_tokenizer = tokenizer

    def create_icl_inference_data(self, training_data, validation_prompts, validation_references):
        prompts = []
        references = []
        pool_embeddings = pool_embeddings = torch.stack([self.get_embeddings(p) for p in training_data])

        for j in tqdm(range(len(validation_prompts))):
            prompt_j = validation_prompts[j]
            nearest_indices, pool_embeddings = self.find_nearest_neighbors(pool_embeddings=pool_embeddings, query=prompt_j)
            prompt = ""
            for i, n in enumerate(nearest_indices):
                prompt += f"Example {i+1}:\n{training_data[n]}\n"
            
            prompt += prompt_j
            reference = validation_references[j]

            prompts.append(prompt)
            references.append(reference)
        return prompts, references

    def get_embeddings(self, prompt):
        encoded_input = self.embedding_tokenizer(prompt, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).squeeze()
        return sentence_embeddings

    def find_nearest_neighbors(self, query, pool=None, pool_embeddings=None, return_sim=False, k=5):
        # Get embeddings for pool and query
        if pool_embeddings == None:
            pool_embeddings = torch.stack([self.get_embeddings(p) for p in pool])
        if type(query) is list:
            query_embedding = torch.stack([self.get_embeddings(q) for q in query])
        else:
            query_embedding = torch.stack([self.get_embeddings(query)])
        similarities = cosine_similarity(query_embedding.cpu(), pool_embeddings.cpu())

        if return_sim:
            return similarities

        nearest_indices = similarities.argsort()[0][-k:][::-1]
        return nearest_indices, pool_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp', default='fl_mod_dep', type=str)
    args = parser.parse_args()
    # from utils import *
    # inf_icl = InferenceICL(model, tokenizer)
    # inf_icl.inference_icl(args.sp)