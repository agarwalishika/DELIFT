import random

class RandomSubsetCreation():
    def __init__(self) -> None:
        pass

    def create_subset(self, n, k=0.3):
        """
        Creates a random subset.

        Args:
            n: total number of possible points
            k: percentage of subset
        Returns:
            Indicies that make up the subset.
        """
        k = int(k * n)
        idx = random.sample(range(0, n), k)

        subset = []
        for i in idx:
            subset.append((i, 0))
        
        return subset