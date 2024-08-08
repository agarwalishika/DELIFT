import numpy as np
import numba as nb
#from annoy import AnnoyIndex
import faiss
from scipy.sparse import lil_matrix
from multiprocessing import Pool, cpu_count

@nb.njit
def calculate_metric_nb(a, b, metric, norm_a=None, norm_b=None, kw=0.1):
    """
    Calculate the specified metric between two matrices a and b.

    Parameters:
    a (np.ndarray): First matrix.
    b (np.ndarray): Second matrix.
    metric (str): The metric to use ('cosine', 'dot', 'euclidean').
    norm_a (np.ndarray, optional): Norm of each row in a. Required for cosine similarity.
    norm_b (np.ndarray, optional): Norm of each row in b. Required for cosine similarity.
    kw (float, optional): Kernel width for rbf metric.
    Returns:
    np.ndarray: The calculated metric between each row of a and b.
    """
    if metric == 'cosine':
        if norm_a is None or norm_b is None:
            raise ValueError("Norms must be provided for cosine similarity.")
        # Reshape norms to ensure proper broadcasting
        a_norm = a if norm_a == 1 else a / norm_a.reshape(-1, 1)
        b_norm = b if norm_b == 1 else b / norm_b.reshape(-1, 1)
        return a_norm @ b_norm.T
    elif metric == 'dot':
        return a @ b.T
    elif metric == 'rbf':
        dist = np.linalg.norm(a[:, None] - b, axis=2)
        sq_dist_mat = dist ** 2
        avg_dist = np.mean(sq_dist_mat)
        np.divide(sq_dist_mat, kw * avg_dist, out=sq_dist_mat)
        np.exp(-sq_dist_mat, out=sq_dist_mat)
        return sq_dist_mat
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_pairwise_in_batches(array1, array2=None, batch_size=1000,
                                metric='cosine', scaling=None, n_jobs=None, kw=0.1):
    """
    Compute pairwise metrics between rows of two arrays in batches.

    Parameters:
    array1 (np.ndarray): First matrix.
    array2 (np.ndarray, optional): Second matrix. If None, uses array1.
    batch_size (int): Size of each batch for processing.
    metric (str): The metric to use ('cosine', 'dot', 'euclidean').
    scaling (str, optional): Type of scaling to apply ('min-max', 'additive').
    n_jobs (int, optional): Number of parallel jobs to run. Defaults to CPU count.
    kw (float, optional): Kernel width for rbf metric.

    Returns:
    np.ndarray: Matrix of calculated pairwise metrics.
    """
    if array2 is None:
        array2 = array1
    n_samples1, n_samples2 = array1.shape[0], array2.shape[0]
    results = np.zeros((n_samples1, n_samples2))
   
    # Evenly distribute batches
    n_batches1 = n_samples1 // batch_size + (n_samples1 % batch_size != 0)
    n_batches2 = n_samples2 // batch_size + (n_samples2 % batch_size != 0)
    array1_batches = np.array_split(array1, n_batches1)
    array2_batches = np.array_split(array2, n_batches2)

    if metric == 'cosine':
        norm1 = np.linalg.norm(array1, axis=1)
        norm2 = np.linalg.norm(array2, axis=1)
        norm1_batches = np.array_split(norm1, n_batches1)
        norm2_batches = np.array_split(norm2, n_batches2)
    else:
        norm1_batches = [1] * n_batches1
        norm2_batches = [1] * n_batches2

    if n_jobs is None or n_jobs < 1:
        n_jobs = cpu_count()

    # Parallel processing for larger datasets
    if n_samples1 * n_samples2 > 10000:  # threshold for using parallel processing
        with Pool(n_jobs) as pool:
            batch_args = [
                (a, b, metric, n1, n2, kw) 
                for a, n1 in zip(array1_batches, norm1_batches) 
                for b, n2 in zip(array2_batches, norm2_batches)
            ]
            batch_results = pool.starmap(calculate_metric_nb, batch_args)

            # Calculate the sizes of each batch
            row_sizes = [a.shape[0] for a in array1_batches]
            col_sizes = [b.shape[0] for b in array2_batches]

            # Initialize accumulators for row and column indices
            row_indices = [0] + row_sizes
            col_indices = [0] + col_sizes

            # Convert to cumulative sums to get start and end indices for each batch
            row_indices = np.cumsum(row_indices)
            col_indices = np.cumsum(col_indices)

            # Iterate through the batch results
            for i, batch_result in enumerate(batch_results):
                row_batch_idx = i // len(array2_batches)
                col_batch_idx = i % len(array2_batches)

                row_start = row_indices[row_batch_idx]
                row_end = row_indices[row_batch_idx + 1]
                col_start = col_indices[col_batch_idx]
                col_end = col_indices[col_batch_idx + 1]

                results[row_start:row_end, col_start:col_end] = batch_result
    else:
        # Single-threaded processing for smaller datasets
        row_accumulator = 0
        for i, a in enumerate(array1_batches):
            col_accumulator = 0
            for j, b in enumerate(array2_batches):
                row_start = row_accumulator
                row_end = row_start + a.shape[0]
                col_start = col_accumulator
                col_end = col_start + b.shape[0]
                results[row_start:row_end, col_start:col_end] = \
                    calculate_metric_nb(a, b, metric, norm1_batches[i], norm2_batches[j], kw)
                col_accumulator += b.shape[0]
            row_accumulator += a.shape[0]

    # Apply scaling if specified
    if scaling == 'min-max':
        min_val, max_val = results.min(), results.max()
        results = (results - min_val) / (max_val - min_val)
    elif scaling == 'additive':
        results = (results + 1) / 2
    return results


#@nb.njit
def convert_distances_to_similarities(distances, metric, kw=0.1):
    """
    Convert distances to similarities based on the specified metric.
    
    Parameters:
    distances (np.ndarray): Matrix of distances.
    metric (str): The metric to use ('cosine', 'rbf').
    kw (float): Kernel width for rbf metric.
    """
    # Compute similarities from distances  
    if metric == 'cosine':
        # For cosine, similarity is 1 - distance
        similarities = 1 - ( (distances ** 2)/2)
    elif metric == 'rbf':
        # For rbf, an example conversion could be using the exponential decay
        # Get squared distance matrix
        similarities = distances ** 2
        avg_dist = np.mean(similarities)
        # Compute the RBF kernel in-place on the squared distance matrix
        np.divide(similarities, kw * avg_dist, out=similarities)
        np.exp(-similarities, out=similarities)
    else:
        raise ValueError(f"Unknown metric for sparse similarity: {metric}")
    return similarities

def compute_similarity_chunk(chunk, knn, num_neighbors, metric, kw, scaling):
    """
    Compute similarities for a chunk of data.

    Parameters:
    chunk (np.ndarray): A chunk of the array1.
    knn (faiss.IndexFlatL2): KNN Index object used for search
    num_neighbors (int): Number of neighbors to find for each row in chunk.
    metric (str): The metric to use.
    kw (float): Kernel width for rbf metric.
    scaling (str, optional): Type of scaling to apply.

    Returns:
    tuple: Indices of neighbors, calculated similarities.
    """
    # Assume AnnoyIndex is built outside and passed as a parameter
    distances, indices = knn.search(chunk, num_neighbors)
    similarities = convert_distances_to_similarities(distances, metric, kw)

    # Apply scaling if specified
    if scaling == 'min-max':
        min_val, max_val = similarities.min(), similarities.max()
        similarities = (similarities - min_val) / (max_val - min_val)
    elif scaling == 'additive':
        similarities = (similarities + 1) / 2

    return indices, similarities

def compute_pairwise_sparse(array1, array2, num_neighbors, metric='cosine', scaling=None, 
                            n_jobs=None, kw=0.1, n_list=100, use_inverse_index=False):
    """
    Compute pairwise similarities between rows of two arrays using sparse representation.

    Parameters:
    array1 (np.ndarray): First matrix.
    array2 (np.ndarray): Second matrix.
    num_neighbors (int): Number of neighbors to find for each row in array1.
    metric (str): The metric to use ('cosine', 'dot', 'euclidean').
    scaling (str, optional): Type of scaling to apply.
    n_jobs (int, optional): Number of parallel jobs for computation.
    kw (float, optional): Kernel width for rbf metric.
    n_list (int, optional): Number of trees for Annoy index.
    use_inverse_index (bool, optional): If True, use inverse index for FAISS
    """
    
    # Normalization for 'cosine' metric
    if metric == 'cosine':
        norm1 = np.linalg.norm(array1, axis=1)
        norm2 = np.linalg.norm(array2, axis=1)
        array1 = array1 / norm1.reshape(-1, 1)
        array2 = array2 / norm2.reshape(-1, 1)

    print(n_jobs)
    if n_jobs is None:
        n_jobs = cpu_count()
        print(n_jobs)

    # Use FAISS for approximate KNN Search
    if use_inverse_index:
        knn_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(array2.shape[1]), 
                                       array2.shape[1], 
                                       n_list, 
                                       faiss.METRIC_L2)
        knn_index.train(array2)
        knn_index.add(array2)
        knn_index.nprobe = 10
    else:
        knn_index = faiss.IndexFlatL2(array2.shape[1])
        knn_index.add(array2)

    chunk_size = 100000
    similarity = lil_matrix((array1.shape[0], array2.shape[0]))

    with Pool(n_jobs) as pool:
        results = []
        for i in range(0, array1.shape[0], chunk_size):
            chunk = array1[i:i + chunk_size]
            result = pool.apply_async(compute_similarity_chunk, (chunk, knn_index, num_neighbors, 
                                                                 metric, kw, scaling))
            results.append((i, result))

        for i, result in sorted(results, key=lambda x: x[0]):
            indices, sim = result.get()
            row_start = i
            row_end = min(i + chunk_size, array1.shape[0])
            row_indices = np.arange(row_start, row_end).repeat(num_neighbors)
            col_indices = indices.reshape(-1)
            similarity[row_indices, col_indices] = sim

    return similarity.tocsr()

class ModelIndependentICLUtility:
    def __init__(self):
        pass

    def compute_pairwise_similarities(self, array1, array2=None, sparse=False, num_neighbors=5, metric='cosine', 
                        batch_size=1000, n_jobs=None, scaling=None, kw=0.1, n_list=100, use_inverse_index=True):
        """
        Compute pairwise similarities between rows of two arrays, either using dense or sparse representation.

        Parameters:
        array1 (np.ndarray): First matrix.
        array2 (np.ndarray, optional): Second matrix. If None, uses array1.
        sparse (bool): If True, use sparse computation. Otherwise, use dense computation.
        num_neighbors (int): Number of neighbors (used in sparse computation).
        metric (str): The metric to use ('cosine', 'dot', 'euclidean').
        batch_size (int): Size of each batch for dense computation.
        n_jobs (int, optional): Number of parallel jobs for computation.
        scaling (str, optional): Type of scaling to apply in dense computation.
        kw (float, optional): Kernel width for rbf metric.
        n_list (int, optional): Number of trees for FAISS index.
        use_inverse_index (bool, optional): If True, use inverse index for FAISS.

        Returns:
        np.ndarray or csr_matrix: Matrix representing pairwise similarities.
        """
        if sparse:
            return compute_pairwise_sparse(array1, array2, num_neighbors, metric, scaling, n_jobs,
                                            kw, n_list, use_inverse_index)
        else:
            return compute_pairwise_in_batches(array1, array2, batch_size, metric, scaling, n_jobs, kw)
