#abstract class for readers to read data from different sources
class DataReader:
    def __init__(self, source):
        self.source = source

    def open(self):
        pass

    def close(self):
        pass

    def read(self):
        pass