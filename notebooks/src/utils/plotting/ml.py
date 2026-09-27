import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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



def plot_confusion_matrix(y_true, y_pred, class_names, figsize=8, fontsize=12, colorbar=False):
    
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45, colorbar=colorbar)
    for text in disp.text_.ravel():
        text.set_fontsize(fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_title("CIFAR-10 Test Set Confusion Matrix", fontsize=fontsize)
    ax.set_xlabel("Predicted label", fontsize=fontsize)
    ax.set_ylabel("True label", fontsize=fontsize)
    plt.tight_layout()
    plt.show()