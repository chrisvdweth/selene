import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from IPython.display import HTML

from itertools import cycle



def plot_data(X, labels=None, centroids=None, sse=None, show_links=True, repeat=False):

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'teal', 'olive', 'maroon']
    color_cycle = cycle(colors)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.axis('equal')
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    def update(frame, l, c, s):
        ax.clear()
        if frame == -1:
            labels = l
            centroids = c
            see = s
        else:
            labels = l[frame]
            centroids = c[frame]
            sse = s[frame]
        color_cycle = cycle(colors)
        if labels is not None:
            for cluster_id in np.unique(labels):
                cluster_sample_indices = np.where(labels == cluster_id)[0]
                X_cluster = X[cluster_sample_indices]
                if X_cluster.shape[0] > 0:
                    ax.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color=next(color_cycle), s=50, alpha=0.6)
        
                    if centroids is not None and show_links is True:
                        for x in X_cluster:
                            ax.plot([x[0], centroids[cluster_id][0]], [x[1], centroids[cluster_id][1]], '--', linewidth=0.5, color='k'.format(cluster_id))
        else:
            plt.scatter(X[:,0], X[:,1], c="gray", s=50, alpha=0.6)
    
        color_cycle = cycle(colors)
        if centroids is not None:
            for centroid in centroids:
                ax.scatter(centroid[0], centroid[1], marker='+', color="black", s=250, lw=5)

        if frame >= 0:
            if frame % 2 == 0:
                ax.set_title(f"Assignment (SSE = {sse})")
            else:
                ax.set_title(f"Update (SSE = {sse})")
        else:
            if s is not None:
                ax.set_title(f"SSE = {s}")

    if labels is not None and labels.ndim == 2:
        anim = animation.FuncAnimation(fig, update, frames=len(labels), fargs=(labels, centroids, sse), interval=1000, repeat=repeat)
        #plt.tight_layout()
        plt.close(fig)
        return HTML(anim.to_html5_video())
    else:
        update(-1, labels, centroids, sse)
        plt.tight_layout()
        plt.show()


def plot_sse_history(sse_values):
    plt.figure()
    x = np.arange(len(sse_values))
    x = x+1
    x = x/2
    x = x + 0.5
    plt.xlabel('Assignment/Update Step', fontsize=14)
    plt.ylabel('SSE', fontsize=18)
    plt.plot(x, sse_values, '-o', lw=3, color="blue", alpha=0.6)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    
    for idx, val in zip(x, sse_values):
        if idx == np.round(idx):
            plt.annotate('A', xy=(idx,val), fontsize=16)
        else:
            plt.annotate('U', xy=(idx,val), fontsize=16)
    
    #plt.tight_layout()
    plt.show()


def plot_elbow(sse_data):
    plt.figure()
    plt.xlabel('K', fontsize=16)
    plt.ylabel('SSE', fontsize=18)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if sse_data.ndim == 2:
        plt.errorbar([s[0] for s in sse_data], [s[1] for s in sse_data], [s[2] for s in sse_data], marker='o', capsize=3, lw=2, color="blue", alpha=0.6)
    else:
        plt.plot(sse_data, marker='o', lw=2, color="blue", alpha=0.6)
    plt.tight_layout()
    plt.show()