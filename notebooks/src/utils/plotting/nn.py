import numpy as np
import matplotlib.pyplot as plt


def plot_training_results(results, legend, normalize=True, fontsize=14):
    results = np.asarray(results)
    plt.figure()
    # Create x range based on the number of results
    x = list(range(1, len(results)+1))

    for idx in range(results.shape[1]):
        y = results[:,idx]
        # Normalize losses so they match the scale in the plot (we are only interested in the trend of the losses!)
        if normalize == True and np.max(y) > 1:
            y = y/np.max(y)
        # Add line to plot
        plt.plot(x, y, lw=3)

    plt.gca().set_xticks(x)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Epoch", fontsize=fontsize)
    plt.legend(legend, loc='lower left', fontsize=fontsize)
    plt.tight_layout()
    plt.show()
