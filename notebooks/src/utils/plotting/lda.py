import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from IPython.display import HTML


def plot_data_2d(X, y, show_means=False, x_label="$x_1$", y_label="$x_2$", s=50, alpha=0.3, ylim=None):
    classes = np.unique(y)
    colors = list(mcolors.BASE_COLORS.keys())
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box') 
    ax.tick_params(labelsize=14)

    for c in classes:
        ax.scatter(X[:,0][np.where(y==c)[0]], X[:,1][np.where(y==c)[0]], color=colors[c], s=s, alpha=alpha)

    if show_means is True:
        overall_mean = np.mean(X, axis=0)
        ax.scatter(overall_mean[0], overall_mean[1], marker='x', color='purple', s=1000, lw=3)
        # Class means
        for c in classes:
            mean = np.mean(X[np.where(y==c)[0]], axis=0)
            ax.scatter(mean[0], mean[1], marker='+', color='black', s=1000, lw=3)
    
    if ylim is not None:
        ax.set_ylim(-ylim, ylim)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)        
    plt.tight_layout()
    plt.show()


def plot_data_3d(X, y, show_means=False, x_label="$x_1$", y_label="$x_2$", z_label="$x_3$", s=50, alpha=0.3, elev=20, azim=None, repeat=False):
    classes = np.unique(y)
    colors = list(mcolors.BASE_COLORS.keys())
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_aspect('equal', adjustable='box') 
    ax.tick_params(labelsize=14)

    for c in classes:
        ax.scatter(X[:,0][np.where(y==c)[0]], X[:,1][np.where(y==c)[0]], X[:,2][np.where(y==c)[0]], color=colors[c], s=s, alpha=alpha)

    if show_means is True:
        overall_mean = np.mean(X, axis=0)
        ax.scatter(overall_mean[0], overall_mean[1], overall_mean[2], marker='x', color='purple', s=1000, lw=3)
        # Class means
        for c in classes:
            mean = np.mean(X[np.where(y==c)[0]], axis=0)
            ax.scatter(mean[0], mean[1], mean[2], marker='+', color='black', s=1000, lw=3)

    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_zlabel(z_label, fontsize=18)
    
    def update(frame):
        ax.view_init(elev=elev, azim=frame)
        return []

    if azim is None:
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, repeat=repeat)
        plt.tight_layout()
        plt.close(fig)
        return HTML(anim.to_html5_video())
    else:
        update(azim)
        plt.tight_layout()            
        plt.show()


def plot_explained_variance(values, style="bar", show_ratios=False):
    if len(values) == 0:
        raise ValueError("values must contain at least one element")

    max_val = np.max(values)
    
    xs = range(len(values))

    fig, ax = plt.subplots(figsize=(6, 4))

    if style == "line":
        ax.plot(xs, values)
        if show_ratios is True:
            for x, value in zip(xs, values):
                ax.text(x, value, f"{value*100:.3g}%", ha="center", va="bottom", fontsize=12)
    elif style == "bar":
        bars = ax.bar(xs, values)
        if show_ratios is True:
            # Add value labels above each bar
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value*100:.3g}%", ha="center", va="bottom", fontsize=12)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{str(i+1)}" for i in xs])
    ax.set_xlabel("Discriminant", fontsize=14)
    ax.set_ylabel("Explained Discriminative Information", fontsize=12)
    ax.set_ylim(0, max_val*1.2)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    plt.show()