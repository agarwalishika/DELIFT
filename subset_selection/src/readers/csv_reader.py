from .reader import DataReader
import csv

class CSVReader(DataReader):
    def __init__(self, source, delimiter=','):
        super().__init__(source)
        self.delimiter = delimiter
        self.file = None
        self.reader = None

    def open(self):
        try:
            self.file = open(self.source, 'r')
            self.reader = csv.DictReader(self.file, delimiter=self.delimiter)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.source}' not found.")
        
    def close(self):
        if self.file:
            self.file.close()

    def read(self):
        if self.file:
            try:
                return next(self.reader)                
            except StopIteration:
                raise None
        else:
            raise ValueError(f"File '{self.source}' is not open. Call open() first.")