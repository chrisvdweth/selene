import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from matplotlib import cm

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

color_cycle = cycle('bgrcmyk')


def plot_sigmoid():
    plt.figure()
    x = np.arange(-5, 5, 0.1)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y, color="orange", lw=2)
    plt.xlabel(r"$z_i$", fontsize=14)
    plt.ylabel(r"$\sigma(z_i)$", fontsize=14)    
    plt.tight_layout()
    plt.show()

    
def plot_csi_data(x, y, y_pred=None, w=None):
    colors = ['blue', 'red']
    plt.figure()
    plt.tick_params(labelsize=14)
    if w is not None:
        x_val = np.arange(27, 35, 0.1)
        # Create artificial feature x0 (all values 1) for bias w0
        x0 = np.ones(x_val.shape[0])
        # Add x0 to initial data matrix
        x_val_bias = np.vstack([x0, x_val]).T
        # Calculate linear signal
        z = np.dot(x_val_bias, w)
        # Calculate prediction output/probability
        y_hat = 1 / (1 + np.exp(-z))
        plt.plot(x_val, y_hat, color='orange', lw=3)    
    if y_pred is None:
        plt.scatter(x, y, color='blue', alpha=0.6, label="data points", s=100)
    else:
        for label in np.unique(y_pred):
            indices = np.argwhere(y_pred==label)
            plt.scatter(x[indices], y[indices], color=colors[label], alpha=0.6, label=f"Data Points (Class {label})", s=100)
    plt.xlabel('Shoe print size', fontsize=16)
    plt.ylabel('P(man)', fontsize=16)
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.show()


def plot_loss_function(X, y, w0_range, w1_range, loss_func):
    Xs, Ys = np.meshgrid(w0_range, w1_range)
    Zs = np.zeros_like(Xs)
    for i0, w0 in enumerate(w0_range):
        for i1, w1 in enumerate(w1_range):
            loss = loss_func(X, y, [w0, w1])
            Zs[i1,i0] = loss
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(30, -15)
    ax.set_xlabel(r'$w_1$', fontsize=16)
    ax.set_ylabel(r'$w_0$', fontsize=16)
    ax.set_zlabel('L', fontsize=16)
    surf = ax.plot_surface(Xs, Ys, Zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.tick_params(labelsize=10)
    plt.tight_layout()
    plt.show()


def plot_ce_loss():
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 4))
    x = np.linspace(0.001, 0.999, 1000)
    y1 = -np.log(x)
    y0 = -np.log(1-x)
    ax1.plot(x, y1, color='orange', lw=3)
    ax2.plot(x, y0, color='orange', lw=3)
    ax1.set_title(r"$y_i = 1$")
    ax2.set_title(r"$y_i = 0$")
    ax1.set_xlabel(r"$\hat{y}_{i}$", fontsize=14)
    ax2.set_xlabel(r"$\hat{y}_{i}$", fontsize=14)
    ax1.set_ylabel(r"$L_{CE}$", fontsize=14)
    ax2.set_ylabel(r"$L_{CE}$", fontsize=14)
    plt.show()


def plot_random_losses(xs, ys, zs):
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(30, -15)
    ax.set_xlabel(r'$w_1$', fontsize=16)
    ax.set_ylabel(r'$w_0$', fontsize=16)
    ax.set_zlabel('L', fontsize=16)
    ax.scatter(xs, ys, zs)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = np.asarray(classes)
    
    if not title:
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), fontsize=14, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
