from abc import ABC, abstractmethod

class Writer(ABC):
    def __init__(self, file_name):
        self.file_name = file_name

    @abstractmethod
    def write_data(self, data):
        """
        Abstract method to write data to a file.
        """
        pass

    def close(self):
        """
        Close the file when done writing.
        """
        pass