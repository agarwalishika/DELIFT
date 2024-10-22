from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import sys
sys.path.append('.')

class Models:
    def __init__(self, embedding_model_name="BAAI/bge-large-en-v1.5", language_model_name="microsoft/Phi-3-mini-4k-instruct", sentence_model_name="paraphrase-MiniLM-L6-v2"):
        """
        Since multiple files requires model and tokenizer objects, instead of creating multiple instances in each file, we can use the Models class to keep the same instances across all the files.
        """

        # define embedding models for ICL
        self.embedding_model_name = embedding_model_name
        self.model_name = language_model_name
        self.sentence_model_name = sentence_model_name

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to('cpu')
        self.embedding_model.eval()

        if self.embedding_tokenizer.pad_token is None:
            self.embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.embedding_model.resize_token_embeddings(len(self.embedding_tokenizer))


        # define language models for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )

        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name, 
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )

        # self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)#, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to('cpu')
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.language_model = torch.nn.DataParallel(self.language_model)
        
        ## left padding for generation
        self.language_tokenizer = AutoTokenizer.from_pretrained(language_model_name, padding_side='left')
        self.language_tokenizer.pad_token = self.language_tokenizer.eos_token