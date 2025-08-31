import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap




def plot_cart_data(X, y, model=None,resolution = 0.005):
    plt.figure(figsize=(5,4))
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # If a model is provided, plot corresponding decision boundaries
    if model is not None:
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
        Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # Plot training samples
    for c in np.unique(y):
        x1 = X[y[:] == c][:,0]
        x2 = X[y[:] == c][:,1]
        plt.scatter(x1, x2, c=colors[int(c)], s=100)
    # Remove all tcs and labels; not important for this plot
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)   
    plt.tight_layout()
    plt.show()


def plot_decision_tree(model):
    plt.figure(figsize=(8,8))
    tree.plot_tree(model, filled = True);
    plt.show()