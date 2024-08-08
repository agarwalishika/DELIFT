#Base class for the sentence encoders
class BaseEncoder:
    def __init__(self, model_name, device, batch_size=512, tokenizer = False):
        if tokenizer:
            self.tokenizer, self.model = self.initialize_model(model_name, device)
        else:
            self.model = self.initialize_model(model_name, device)
        self.device = device
        self.batch_size = batch_size

    def initialize_model(self, model_name, device):
        pass

    def encode(self, inputs, return_tensors=False):
        pass