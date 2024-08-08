import json
from .writer import Writer

class JsonLWriter(Writer):
    def __init__(self, file_name):
        super().__init__(file_name)

    def write_data(self, data):
        """
        Write a JSON object to a new line in the JSON Lines file.
        """
        with open(self.file_name, 'a') as file:
            json.dump(data, file)
            file.write('\n')

    def close(self):
        """
        Close the file when done writing.
        """
        pass