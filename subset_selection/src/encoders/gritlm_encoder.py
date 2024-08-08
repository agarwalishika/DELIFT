from gritlm import GritLM
from .base_encoder import BaseEncoder

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def encode_with_gritlm(inputs, model_name = "GritLM/GritLM-7B", instruction=""):
    # Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head)
    lm = GritLM(model_name, torch_dtype="auto", mode="embedding")
    # Encode the inputs
    encodings = lm.encode(inputs, instruction=gritlm_instruction(instruction))
    return encodings

class GritLMEncoder(BaseEncoder):
    def initialize_model(self, model_name ="GritLM/GritLM-7B"):
        return GritLM(model_name, torch_dtype="auto", mode="embedding")

    def encode(self, inputs, instruction=""):
        return self.model.encode(inputs, instruction=gritlm_instruction(instruction))
        