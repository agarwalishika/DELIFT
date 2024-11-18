import sys
sys.path.append('.')
import subset_selection.model_inference as mi
from sklearn.cluster import KMeans
import numpy as np
from numpy.linalg import norm

class DEFTSubsetCreation():
    def __init__(self, models) -> None:
        self.models = models

    def create_subset(self, data, k=0.3, alpha=0.5, n_clusters=7):
        """
        Creates a subset with DEFT-UCS.

        Args:
            data: the data
            k: percentage of subset
            alpha: easy samples to use
            beta: hard samples to use
        Returns:
            Indicies that make up the subset.
        """

        # embed data
        k = int(k * len(data) / n_clusters)
        embeddings = mi.batch_inference(self.models.embedding_model, self.models.embedding_tokenizer, list(data)).cpu()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(embeddings)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        subset = []
        for cluster_num in range(n_clusters):
            distances = []
            for i, l in enumerate(labels):
                if l != cluster_num:
                    continue
                dist = np.dot(embeddings[i], centroids[cluster_num])
                if dist > 0:
                    distances.append(dist)

            distances_argsorted = np.argsort(np.array(distances))
            subset.extend(distances_argsorted[:int(alpha * k)])
            subset.extend(distances_argsorted[int((alpha-1) * k):])

        subset_aug = []
        for x in subset:
            subset_aug.append((x, 0))

        return subset_aug