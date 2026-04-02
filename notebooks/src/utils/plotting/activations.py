import numpy as np
import matplotlib.pyplot as plt


def plot_example_data(X, y, y_pred=None, y_pred_label='', title=''):
    # Plot the data
    plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title)
    #plt.scatter(x, y, s=10, label=labels[0], c="blue", alpha=0.5)
    plt.scatter(X, y, color="blue", alpha=0.25, label="Data Points")
    if y_pred is not None:
        plt.plot(X, y_pred, label=y_pred_label, color="red", linewidth=3)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.show()



def plot_activation_function(f, df, label=''):
    x = np.linspace(-6, 6, 200)
    y = f(x)
    dy = df(x)
    plt.figure(figsize=(8, 5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.plot(x, y, label=label, color="blue", linewidth=3)
    plt.plot(x, dy, label=f"{label} (Derivative)", color="red", linewidth=3, linestyle="--")
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.show()