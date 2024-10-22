from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch

class InferenceICL():
    def __init__(self, model, tokenizer) -> None:
        self.embedding_model = model
        self.embedding_tokenizer = tokenizer

    def create_icl_inference_data(self, training_data, validation_prompts, validation_references, k=5):
        """
        Adds the k-most semantically similar examples from the training set to each point in the validation set as in-context examples.

        Args:
            training_data: set of training prompts and references (as in DataObject)
            validation_prompts: set of validation prompts (as in DataObject)
            validation_references: set of validation references (as in DataObject)
        Returns:
            List of augmented prompts and references.
        """
        prompts = []
        references = []
        pool_embeddings = pool_embeddings = torch.stack([self.get_embeddings(p) for p in training_data])

        for j in tqdm(range(len(validation_prompts))):
            prompt_j = validation_prompts[j]
            nearest_indices, pool_embeddings = self.find_nearest_neighbors(pool_embeddings=pool_embeddings, query=prompt_j, k=k)
            prompt = ""
            for i, n in enumerate(nearest_indices):
                prompt += f"Example {i+1}:\n{training_data[n]}\n"
            
            prompt += prompt_j
            reference = validation_references[j]

            prompts.append(prompt)
            references.append(reference)
        return prompts, references

    def get_embeddings(self, prompt):
        """
        Embeds the given prompt using an embedding model.

        Args:
            prompt: singular prompt or list of prompts to embed
        Returns:
            Prompt embedding
        """
        self.embedding_model = self.embedding_model.to('cuda')
        encoded_input = self.embedding_tokenizer(prompt, padding=True, truncation=True, return_tensors='pt').to(self.embedding_model.device)
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).squeeze()
        self.embedding_model.to('cpu')
        return sentence_embeddings

    def find_nearest_neighbors(self, query, pool=None, pool_embeddings=None, return_sim=False, k=5):
        """
        Given a query (singular or list), return the nearest neighbors wrt cosine simliarity in the pool set.

        Args:
            query: singular prompt or list of prompts to search for
            pool: list of prompts to search from (or provide pool_embeddings)
            pool_embeddings: embeddings of the list of prompts to search from (or provide pool)
            return_sim: True if just the similarity values should be returned, False if the indices and pool_embeddings should be returned
            k: number of nearest neighbors
        Returns:
            Either the similarity values, or the indices/pool_embeddings (for future use)
        """
        assert not (pool == None and pool_embeddings == None)

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