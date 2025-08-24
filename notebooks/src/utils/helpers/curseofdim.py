import numpy as np

def compute_pairwise_distances(A):
    sq_norms = np.sum(A**2, axis=1)                                      # (n,)
    dists_squared = sq_norms[:, None] + sq_norms[None, :] - 2 * A @ A.T  #
    dists_squared = np.maximum(dists_squared, 0)                         # For numerical stability
    distances = np.sqrt(dists_squared)                                   # 
    np.fill_diagonal(distances, np.nan)                                  # Mask diagonal with nan values to ignore distance between a vector and itself
    distances = distances.flatten()                                      # Flatten matrix of pairwise distance to 1d array
    return distances[~np.isnan(distances)]                               # Remove nan values and return array