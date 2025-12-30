from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def cluster_kmeans(features, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(features)
    return labels

def plot_tsne(features, labels, fname):
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)  # <-- FIX: ensure folder exists

    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=5, cmap="tab10")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
