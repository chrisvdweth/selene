import numpy as np
import pandas as pd
import string
import sys
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import euclidean_distances
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from itertools import cycle


def plot_distance_matrix(matrix, clusters, precision=2):
    matrix = np.round(matrix, decimals=precision)
    
    labels = np.asarray(list(string.ascii_uppercase))
    columns = [ ''.join(labels[c]) for c in clusters ]
    
    df = pd.DataFrame(matrix, columns=columns)
    df.index = columns

    df = df.replace(np.inf, '')
    return df


def plot_linkage_example(X, y, method=None):

    colors = ["blue", "red", "green"]
    plt.figure(figsize=(6, 5))
    plt.gca().tick_params(axis='x', labelsize=14)
    plt.gca().tick_params(axis='y', labelsize=14)
    for i, _ in enumerate(colors):
        cluster_sample_indices = np.where(y == i)[0]
        plt.scatter(X[cluster_sample_indices, 0], X[cluster_sample_indices, 1], c=colors[i], edgecolor='k', s=100, alpha=0.6)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18)
    
    if method is not None and method.lower() in ["single", "complete"]:
        clusters, distances, arrows = [], [], []
        for cid in np.unique(y):
            cluster_sample_indices = np.where(y == cid)[0]
            clusters.append(X[cluster_sample_indices])
        # Nested loop to calculate distances between all pairs of clusters
        for idx1, c1 in enumerate(clusters):
            for idx2, c2 in enumerate(clusters):
                # Ignore values on the diagonal and lower triangual matrix
                if idx1 >= idx2:
                    continue
                min_dist, max_dist, min_p, min_q, max_p, max_q = np.inf, -np.inf, None, None, None, None
                for p in c1:
                    for q in c2:
                        distance = np.linalg.norm(p-q)
                        if distance < min_dist:
                            min_dist = distance
                            min_p, min_q = p, q
                        if distance > max_dist:
                            max_dist = distance
                            max_p, max_q = p, q
                if method.lower() == "single":
                    distances.append(min_dist)
                    arrows.append((tuple(min_p), tuple(min_q)))
                elif method.lower() == "complete":
                    distances.append(max_dist)
                    arrows.append((tuple(max_p), tuple(max_q)))
        
        if method.lower() in ["single", "complete"]:
            for a in arrows:
                plt.annotate(text='', xy=a[0], xytext=a[1], arrowprops=dict(arrowstyle='<->', lw=2, linestyle="dashed", color="gray"))
            idx = np.argmin(distances)
            plt.annotate(text='', xy=arrows[idx][0], xytext=arrows[idx][1], arrowprops=dict(arrowstyle='<->', lw=2, color="black"))
    
    plt.show()


def plot_clusters(X, labels=None, level=None):
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'teal', 'olive', 'magenta', 'maroon', 'cyan']
    color_cycle = cycle(colors)

    letters = list(string.ascii_uppercase)
    letter_cycle = cycle(letters)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.axis('equal')
    plt.margins(x=0.2, y=0.3)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    if labels is None:
        ax.scatter(X[:,0], X[:,1], c="gray", edgecolors='black', s=200, alpha=0.4)
    else:
        for cluster_id in np.unique(labels):
            cluster_sample_indices = np.where(labels == cluster_id)[0]
            X_cluster = X[cluster_sample_indices]
            ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), edgecolors='black', s=200, alpha=0.6)
        if level is not None:
            plt.title(f"Hierarchy Level {level}", fontsize=14)

    for x in X:
        plt.text(x[0]+0.2, x[1]+0.2, next(letter_cycle), fontsize=22)


def plot_data(X):
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.axis('equal')
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$x_2$", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.scatter(X[:,0], X[:,1], c="gray", edgecolors='black', s=100, alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_labels(X, labels=None):

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'teal', 'olive', 'magenta', 'maroon', 'cyan']
    color_cycle = cycle(colors)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.axis('equal')
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    if labels is None:
        plt.scatter(X[:,0], X[:,1], c="gray", edgecolors='black', s=100, alpha=0.4)
    else:
        for cluster_id in np.unique(labels):
            cluster_sample_indices = np.where(labels == cluster_id)[0]
            X_cluster = X[cluster_sample_indices]
            ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), edgecolors='black', s=100, alpha=0.6)
    plt.show()



def plot_linkage_comparison(X, n_clusters=2):

    single   = AgglomerativeClustering(n_clusters=n_clusters, linkage="single").fit(X)
    complete = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete").fit(X)
    average  = AgglomerativeClustering(n_clusters=n_clusters, linkage="average").fit(X)
    ward     = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X)

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'teal', 'olive', 'magenta', 'maroon', 'cyan']
    
    
    # Create a 2x2 grid of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    #plt.axis('equal')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()
    
    ax1.set_title('Single Linkage')
    ax2.set_title('Complete Linkage')
    ax3.set_title('Average Linkage')
    ax4.set_title('Ward Linkage')
    
    color_cycle = cycle(colors)
    for cluster_id in np.unique(single.labels_):
        cluster_sample_indices = np.where(single.labels_ == cluster_id)[0]
        X_cluster = X[cluster_sample_indices]
        ax1.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), edgecolors='black', s=50, alpha=0.6)

    color_cycle = cycle(colors)
    for cluster_id in np.unique(complete.labels_):
        cluster_sample_indices = np.where(complete.labels_ == cluster_id)[0]
        X_cluster = X[cluster_sample_indices]
        ax2.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), edgecolors='black', s=50, alpha=0.6)

    color_cycle = cycle(colors)
    for cluster_id in np.unique(average.labels_):
        cluster_sample_indices = np.where(average.labels_ == cluster_id)[0]
        X_cluster = X[cluster_sample_indices]
        ax3.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), edgecolors='black', s=50, alpha=0.6)

    color_cycle = cycle(colors)
    for cluster_id in np.unique(ward.labels_):
        cluster_sample_indices = np.where(ward.labels_ == cluster_id)[0]
        X_cluster = X[cluster_sample_indices]
        ax4.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), edgecolors='black', s=50, alpha=0.6)

    plt.tight_layout()
    plt.show()



def plot_dendrogram(model, lw=1, **kwargs):

    # Count samples under each non-leaf node
    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0

        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]

        counts[i] = current_count

    linkage_matrix = np.column_stack([
        model.children_,
        model.distances_,
        counts
    ]).astype(float)
    
    with plt.rc_context({'lines.linewidth': lw}):
        dendrogram(linkage_matrix, **kwargs)
    
    plt.xlabel("Sample index or cluster size")
    plt.ylabel("Distance")
    plt.show()