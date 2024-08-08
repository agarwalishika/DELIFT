from .reader import DataReader
from datasets import load_dataset

class HFDatasetReader(DataReader):
    def __init__(self, dataset_name, split):
        super().__init__(dataset_name)
        self.dataset = None
        self.dataset_iter = None
        self.split = split

    def open(self):
        try:
            self.dataset = load_dataset(self.source, self.split)
            # self.dataset_iter = iter(self.dataset)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset '{self.source}' not found.")
    
    def close(self):
        self.dataset = None
        self.dataset_iter = None
    
    def read(self):
        if self.dataset:
            if self.dataset_iter is None:
                self.dataset_iter = iter(self.dataset[self.split])
            try:
                return next(self.dataset_iter)
            except StopIteration:
                return None
        else:
            raise ValueError(f"Dataset '{self.source}' is not open. Call open() first.")