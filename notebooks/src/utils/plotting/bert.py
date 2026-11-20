import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def plot_top_tokens(top_tokens, top_probs, title=None):
    plt.figure(figsize=(7, 5))
    plt.barh(top_tokens[::-1], top_probs.numpy()[::-1])
    plt.xlabel("Probability")
    plt.title(title)
    plt.show()


def plot_heatmap(values, labels, lower=True): 
    
    if lower is True:
        mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    else:
        np.ones_like(values, dtype=bool)
        
    labels = [f"{i+1}: {s[:25]}" for i, s in enumerate(labels)]
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        values,
        mask=mask,
        annot=True, fmt=".2f",
        cmap="Blues",
        vmin=values.min(), vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "Cosine similarity"},
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Sentence similarity (mean-pooled BERT embeddings)")
    plt.tight_layout()
    plt.show()