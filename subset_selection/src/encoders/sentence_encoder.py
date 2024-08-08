from sentence_transformers import SentenceTransformer
from .base_encoder import BaseEncoder

class SentenceEncoder(BaseEncoder):
    def initialize_model(self, model_name, device):
        return SentenceTransformer(model_name, device=device)

    def encode(self, inputs, return_tensors=False, normalize_embeddings=False):
        if return_tensors:
            return self.model.encode(inputs, device=self.device,
                                     batch_size=self.batch_size,
                                     convert_to_tensor=return_tensors, 
                                     show_progress_bar = True,
                                     normalize_embeddings = normalize_embeddings).cpu()
        else:
            return self.model.encode(inputs, device=self.device, 
                                     batch_size=self.batch_size,
                                     convert_to_numpy=True, 
                                     show_progress_bar = True,
                                     normalize_embeddings = normalize_embeddings)