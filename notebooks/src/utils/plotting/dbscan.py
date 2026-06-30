import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from itertools import cycle
from sklearn.neighbors import NearestNeighbors


POINT_TYPE_UKNOWN = 0
POINT_TYPE_CORE   = 1
POINT_TYPE_BORDER = 2
POINT_TYPE_NOISE  = 3



def plot_labels(X, labels=None):

    colors = ['red', 'green', 'orange', 'purple', 'teal', 'olive', 'magenta', 'maroon', 'blue', 'cyan']
    color_cycle = cycle(colors)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.axis('equal')
    ax.set_xlabel("X1", fontsize=18)
    ax.set_ylabel("X2", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    if labels is None:
        plt.scatter(X[:,0], X[:,1], c="gray", edgecolors='black', s=100, alpha=0.4)
    else:
        for cluster_id in np.unique(labels):
            cluster_sample_indices = np.where(labels == cluster_id)[0]
            X_cluster = X[cluster_sample_indices]
            if cluster_id == -1:
                ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color="black", edgecolors='black', s=100, alpha=0.8)
            elif cluster_id >= 0:
                ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), edgecolors='black', s=100, alpha=0.6)
            else:
                ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color="gray", edgecolors='black', s=100, alpha=0.4)    

    plt.show()



def plot_label_history(X, label_history=None, interval=500, repeat=False):

    colors = ['red', 'green', 'orange', 'purple', 'teal', 'olive', 'magenta', 'maroon', 'blue', 'cyan']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.axis('equal')
    ax.set_xlabel("X1", fontsize=18)
    ax.set_ylabel("X2", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    def update(frame):
        ax.clear()
        color_cycle = cycle(colors)
        labels = label_history[frame]
        if labels is None:
            plt.scatter(X[:,0], X[:,1], c="gray", edgecolors='black', s=100, alpha=0.4)
        else:
            for cluster_id in np.unique(labels):
                cluster_sample_indices = np.where(labels == cluster_id)[0]
                X_cluster = X[cluster_sample_indices]
                if cluster_id == -1:
                    ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color="black", edgecolors='black', s=100, alpha=0.8)
                elif cluster_id >= 0:
                    ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), edgecolors='black', s=100, alpha=0.6)
                else:
                    ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color="gray", edgecolors='black', s=100, alpha=0.4)    

    anim = animation.FuncAnimation(fig, update, frames=len(label_history), interval=interval, repeat=repeat)
    plt.close(fig)
    return HTML(anim.to_html5_video())



def plot_types(X, types=None):

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.axis('equal')
    ax.set_xlabel("X1", fontsize=18)
    ax.set_ylabel("X2", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    if types is None:
        plt.scatter(X[:,0], X[:,1], c="gray", edgecolors='black', s=100, alpha=0.4)
    else:
        for point_type in np.unique(types):
            point_type_indices = np.where(types == point_type)[0]
            X_point_type = X[point_type_indices]
            if point_type == POINT_TYPE_CORE:
                ax.scatter(X_point_type[:,0], X_point_type[:,1], marker='o', color="blue", edgecolors='black', s=100, alpha=0.6)
            elif point_type == POINT_TYPE_BORDER:
                ax.scatter(X_point_type[:,0], X_point_type[:,1], marker='o', color="cyan", edgecolors='black', s=100, alpha=0.6)
            elif point_type == POINT_TYPE_NOISE:
                ax.scatter(X_point_type[:,0], X_point_type[:,1], marker='o', color="black", edgecolors='black', s=100, alpha=0.8)
            else:
                ax.scatter(X_point_type[:,0], X_point_type[:,1], marker='o', color="gray", edgecolors='black', s=100, alpha=0.4)    

    plt.show()



def plot_type_history(X, type_history=None, interval=500, repeat=False):

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.axis('equal')
    ax.set_xlabel("X1", fontsize=18)
    ax.set_ylabel("X2", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    def update(frame):
        ax.clear()
        types = type_history[frame]
    
        if types is None:
            plt.scatter(X[:,0], X[:,1], c="gray", edgecolors='black', s=100, alpha=0.4)
        else:
            for point_type in np.unique(types):
                point_type_indices = np.where(types == point_type)[0]
                X_point_type = X[point_type_indices]
                if point_type == POINT_TYPE_CORE:
                    ax.scatter(X_point_type[:,0], X_point_type[:,1], marker='o', color="blue", edgecolors='black', s=100, alpha=0.6)
                elif point_type == POINT_TYPE_BORDER:
                    ax.scatter(X_point_type[:,0], X_point_type[:,1], marker='o', color="cyan", edgecolors='black', s=100, alpha=0.6)
                elif point_type == POINT_TYPE_NOISE:
                    ax.scatter(X_point_type[:,0], X_point_type[:,1], marker='o', color="black", edgecolors='black', s=100, alpha=0.8)
                else:
                    ax.scatter(X_point_type[:,0], X_point_type[:,1], marker='o', color="gray", edgecolors='black', s=100, alpha=0.4)    

    anim = animation.FuncAnimation(fig, update, frames=len(type_history), interval=interval, repeat=repeat)
    plt.close(fig)
    return HTML(anim.to_html5_video())



def plot_k_distance(X, k=4):

    X = np.asarray(X)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (n_samples, 2).")

    if k < 1 or k >= len(X):
        raise ValueError("k must be between 1 and n_samples - 1.")

    # Include the point itself, so we request k + 1 neighbors
    neighbors = NearestNeighbors(n_neighbors=k + 1)
    neighbors.fit(X)

    distances, _ = neighbors.kneighbors(X)

    # Column 0 is distance to itself, so column k is the k-th neighbor
    k_distances = np.sort(distances[:, k])

    plt.figure(figsize=(8, 5))
    plt.gca().tick_params(axis='x', labelsize=14)
    plt.gca().tick_params(axis='y', labelsize=14)
    plt.plot(k_distances, lw=3, color="blue")
    plt.xlabel("Points sorted by k-distance", fontsize=14)
    plt.ylabel(f"Distance to {k}-th nearest neighbor", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()