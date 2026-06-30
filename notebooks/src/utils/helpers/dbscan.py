import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

POINT_TYPE_UKNOWN = 0
POINT_TYPE_CORE   = 1
POINT_TYPE_BORDER = 2
POINT_TYPE_NOISE  = 3


def _get_neighbors(X, x, eps):
    # Compute the Euclidean distance between x and all data points
    distances = euclidean_distances(X, x.reshape(1,-1)).squeeze()
    # Return the indices of the data points that a closer than eps to x
    return np.where(distances <= eps)[0]


def explore_history(X, core_point_idx, eps, min_samples, cluster_id, labels, types, label_history, type_history):
    S = set([core_point_idx])

    while len(S) != 0:
        # Get the next index from S and remove from S
        s = S.pop()

        labels[s] = cluster_id
        types[s]  = POINT_TYPE_CORE
        label_history.append(labels.copy())
        type_history.append(types.copy())
        
        neighbors = _get_neighbors(X, X[s], eps)

        for n in neighbors:
            
            neighbors_of_neighbor = _get_neighbors(X, X[n], eps)

            if len(neighbors_of_neighbor) < min_samples:
                types[n] = POINT_TYPE_BORDER
                labels[n] = cluster_id
                label_history.append(labels.copy())
                type_history.append(types.copy())
            else:
                if types[n] == POINT_TYPE_UKNOWN:
                    S.add(n)



def dbscan_history(X, eps=0.5, min_samples=5):
    # Initialize
    labels, label_history = [-2] * X.shape[0], []
    types, type_history  = [POINT_TYPE_UKNOWN] * X.shape[0], []

    cluster_id =-1

    for idx, x in enumerate(X):
        
        # Check if sample has already been labeled; if Yes, nothing to do
        if types[idx] != POINT_TYPE_UKNOWN:
            continue

        # Get all neighbors of x (within minimum distance eps)
        neighbors = _get_neighbors(X, x, eps)

        # If the number of neighbors is to small, label x as noise
        if len(neighbors) < min_samples:
            labels[idx] = -1
            types[idx]  = POINT_TYPE_NOISE
            label_history.append(labels.copy())
            type_history.append(types.copy())
            continue

        # If we reach this part, we found a new core point x!
        cluster_id = cluster_id + 1

        explore_history(X, idx, eps, min_samples, cluster_id, labels, types, label_history, type_history)

    return labels, types, label_history, type_history
