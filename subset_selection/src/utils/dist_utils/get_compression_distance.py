import numpy as np
import torch

class CompressionDistance:
    """
    Used to calculate the distance between compressed objects.
    """
    def __init__(self, compressed_values_a, compressed_values_b, compressed_values_ab):
        if isinstance(compressed_values_a, np.ndarray):
            self.compressed_values_a = compressed_values_a
            self.compressed_values_b = compressed_values_b
            self.compressed_values_ab = compressed_values_ab
        elif isinstance(compressed_values_a, torch.Tensor):
            self.compressed_values_a = compressed_values_a
            self.compressed_values_b = compressed_values_b
            self.compressed_values_ab = compressed_values_ab
        else:
            self.compressed_values_a = np.array(compressed_values_a)
            self.compressed_values_b = np.array(compressed_values_b)
            self.compressed_values_ab = np.array(compressed_values_ab)

    def _ncd(self):
        if isinstance(self.compressed_values_a, np.ndarray):
            denominator = np.maximum(self.compressed_values_a, self.compressed_values_b)
            numerator = self.compressed_values_ab - np.minimum(self.compressed_values_a, self.compressed_values_b)
            distance = numerator / denominator
        elif isinstance(self.compressed_values_a, torch.Tensor):
            denominator = torch.maximum(self.compressed_values_a, self.compressed_values_b)
            numerator = self.compressed_values_ab - torch.minimum(self.compressed_values_a, self.compressed_values_b)
            distance = numerator / denominator
        else:
            raise ValueError("Unsupported data type")
        return distance