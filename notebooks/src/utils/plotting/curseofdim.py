import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def test_orthogonality(n_dim, n_vecs=1000, min_cos_sim=0.01):
    # Create a random matrix with n_trial vectors of size n_dim
    X = np.random.randn(n_vecs, n_dim)
    # Compute all pairwise cosine similarities
    C = cosine_similarity(X,X)
    # Get the number of cosine similarities lower than the minimum threshold
    num_almost_orthogonal = (C < min_cos_sim).sum()
    # Compute and the ratio
    return (num_almost_orthogonal) / ((n_vecs*n_vecs)-n_vecs)

def plot_pairwise_distance_histogram(pairwise_distances, num_dim):
    max_dist = np.linalg.norm(np.array([1.0]*num_dim) - np.array([0.0]*num_dim))
    plt.figure(figsize=(7, 4.5))
    plt.xlim(0, max_dist)
    plt.ylim(0, 1.8)
    #plt.xticks(fontsize=14)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.yticks(fontsize=14)
    plt.xlabel('Distance', fontsize=16)
    plt.ylabel('Frequency (%)', fontsize=16)
    plt.hist(pairwise_distances, bins=200, density=True, label='#dimensions={}'.format(num_dim))
    plt.legend(loc='upper right', prop={'size': 15})
    plt.tight_layout()

def plot_orthogonalities(x, ys, labels=[]):
    plt.figure(figsize=(7, 4.5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    for idx, y in enumerate(ys):
        plt.plot(x, y, label=f"{labels[idx]}", marker="o", lw=3)
    plt.xlabel("Dimension (size of vectors)", fontsize=14)
    plt.ylabel("Ratio of almost orthogonal vectors", fontsize=13)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

def plot_subspace_sizes(x, y, k_over_N):
    plt.figure(figsize=(7, 4.5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.plot(x, y, label=f"k/N={k_over_N}", marker="o", lw=3)
    plt.xlabel("Dimension (size of vectors)", fontsize=14)
    plt.ylabel("L (size of subspace)", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()    