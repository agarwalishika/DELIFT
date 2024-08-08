from .reader import DataReader
import json

class JsonLReader(DataReader):
    def __init__(self, source):
        super().__init__(source)
        self.file = None

    def open(self):
        try:
            self.file = open(self.source, 'r')
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.source}' not found.")
        return self  # Important: return self for use in the context manager

    def close(self):
        if self.file:
            self.file.close()

    def read(self):
        if self.file:
            try:
                line = self.file.readline()
                if line:
                    return json.loads(line)
                else:
                    return None  # EOF
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Invalid JSON format in file '{self.source}'.")
        else:
            raise ValueError("File is not open. Call open() first.")

    # Context manager methods
    def __enter__(self):
        self.open()
        return self  # This allows the object to be used within a with statement

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()  # Clean up by closing the file