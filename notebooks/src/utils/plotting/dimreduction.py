import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(correlation_matrix, cmap="coolwarm", font_size=12):
    plt.figure()
    with plt.style.context({'axes.labelsize':12, 'xtick.labelsize':font_size, 'ytick.labelsize':font_size}):
        ax = sns.heatmap(correlation_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, square=True, annot=True, annot_kws={'size': font_size})
    plt.show()

def plot_feature_importances(importances, std_devs, feature_names):
    x = np.arange(len(importances))
    plt.figure()
    plt.bar(x, importances, yerr=std_devs, capsize=5)
    plt.xticks(x, feature_names, fontsize=14)
    plt.ylabel("Mean decrease in impurity", fontsize=14)
    plt.show()    

def plot_swiss_roll(X, colors):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=50, alpha=0.8)
    ax.view_init(azim=-66, elev=12)
    plt.tight_layout()
    plt.show()

def plot_2d_projection(X, colors=None, cmap=None):
    plt.figure()
    if colors is None:
        colors = "blue"
    plt.scatter(X[:,0], X[:,1], s=100, clip_on=False, c=colors, alpha=0.6)
    plt.xlabel("$x^\prime$", fontsize=18)
    plt.ylabel("$x^{\prime\prime}$", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.show()