from .reader import DataReader
import json

class JsonReader(DataReader):
    def __init__(self, source):
        super().__init__(source)
        self.file = None

    def open(self):
        try:
            self.file = open(self.source, 'r')
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.source}' not found.")
        
    def close(self):
        if self.file:
            self.file.close()

    def read(self):
        if self.file:
            try:
                return json.load(self.file)
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in file '{self.source}'.")
        else:
            raise ValueError(f"File '{self.source}' is not open. Call open() first.")