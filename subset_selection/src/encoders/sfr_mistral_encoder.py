import token
from torch.utils.data import BatchSampler, SequentialSampler
from .base_encoder import BaseEncoder
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# # Each query must come with a one-sentence instruction that describes the task
# task = 'Given a web search query, retrieve relevant passages that answer the query'

class SFRMistralEncoder(BaseEncoder):
    def __init__(self, model_name='Salesforce/SFR-Embedding-Mistral', 
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                 batch_size=20):
        super().__init__(model_name, device, batch_size, tokenizer=True)

    def initialize_model(self, model_name, device):    
        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        return tokenizer, model

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, inputs, return_tensor=True):

        # get the embeddings
        max_length = 4096
        sampler = BatchSampler(SequentialSampler(range(len(inputs))), 
                               batch_size=self.batch_size, drop_last=False)
        
        batched_tokenized_inputs = []
        for indices in sampler:
            inputs_batch = [inputs[x] for x in indices]
            batched_tokenized_inputs.append(self.tokenizer(inputs_batch, 
                                                           max_length=max_length, 
                                                           padding=True, 
                                                           truncation=True, 
                                                           return_tensors="pt"))
        embeddings = []
        for batch_tokens in batched_tokenized_inputs:
            batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
            with torch.no_grad():
                outputs = self.model(**batch_tokens)
            batch_embeddings = self.last_token_pool(outputs.last_hidden_state, 
                                              batch_tokens['attention_mask']).cpu()
            embeddings.append(batch_embeddings)

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if return_tensor:
            return embeddings
        return embeddings.cpu().numpy()