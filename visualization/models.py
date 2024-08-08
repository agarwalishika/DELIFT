import sys
sys.path.append('.')
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Models:
    def __init__(self, embedding_model_name="BAAI/bge-large-en-v1.5", language_model_name="EleutherAI/gpt-neo-125m", sentence_model_name="paraphrase-MiniLM-L6-v2"):
        """
        Since multiple files requires model and tokenizer objects, instead of creating multiple instances in each file, we can use the Models class to keep the same instances across all the files.
        """

        # define embedding models for ICL
        self.embedding_model_name = embedding_model_name
        self.model_name = language_model_name
        self.sentence_model_name = sentence_model_name

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name, device_map="auto")
        self.embedding_model.eval()

        if self.embedding_tokenizer.pad_token is None:
            self.embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.embedding_model.resize_token_embeddings(len(self.embedding_tokenizer))


        # define language models for inference
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name, device_map="auto")
        ## left padding for generation
        self.language_tokenizer = AutoTokenizer.from_pretrained(language_model_name, padding_side='left')
        self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        # define a semantic similarity model (was used eariler to measure the performance of model inference, but we use ROUGE instead now)
        ## self.sem_sim_model = SentenceTransformer(sentence_model_name)

        self.device = device