#!/usr/bin/env python3
"""
Hard-task clustering evaluation script.

Evaluates multiple representations:
- MFCC -> PCA
- Dense VAE
- Conv VAE
- Beta-Conv VAE
- Conv Autoencoder

Clustering:
- KMeans
- Agglomerative
- DBSCAN

Metrics:
- Silhouette Score
- Davies–Bouldin Index
- Adjusted Rand Index (if genre labels available)

Visuals:
- UMAP embeddings
- Cluster composition plots

All outputs saved to results/hard/
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score
)
from sklearn.decomposition import PCA
import umap

# --------------------------------------------------
# Path setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = Path("results/hard")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Safe filename utilities (Windows-safe)
# --------------------------------------------------
_BAD_CHARS = r'[<>:"/\\|?*\n\r\t]'

def safe_filename(name: str) -> str:
    name = re.sub(_BAD_CHARS, "_", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")

# --------------------------------------------------
# Helper: Check if clustering is valid for metrics
# --------------------------------------------------
def is_valid_clustering(labels, min_clusters=2):
    """Check if labels contain at least min_clusters distinct non-noise clusters."""
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label if present
    return len(unique_labels) >= min_clusters

# --------------------------------------------------
# Plot helpers
# --------------------------------------------------
def umap_plot(X, labels, title, outpath):
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    emb = reducer.fit_transform(X)

    plt.figure(figsize=(6, 5))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=8, cmap="tab10")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def cluster_composition_plot(labels, genres, title, outpath):
    df = pd.DataFrame({"cluster": labels, "genre": genres})
    comp = pd.crosstab(df["cluster"], df["genre"], normalize="index")

    comp.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="tab20")
    plt.title(title)
    plt.ylabel("Proportion")
    plt.xlabel("Cluster")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

# --------------------------------------------------
# Main evaluation
# --------------------------------------------------
def main():
    print("\n=== HARD TASK: CLUSTERING EVALUATION ===\n")

    manifest = pd.read_csv("data/fma_manifest_5k_5genres_lyrics_whisper_dropped_removed.csv")

    if "track_id" not in manifest.columns or "genre" not in manifest.columns:
        raise ValueError("Manifest must contain 'track_id' and 'genre' columns.")

    # Map: track_id -> genre label (ground truth)
    tid_to_genre = dict(zip(manifest["track_id"].astype(int), manifest["genre"]))

    # ------------------------------
    # Load representations + their ID files
    # ------------------------------
    reps = {
        "MFCC->PCA(8)": {
            "X": PCA(n_components=8, random_state=42).fit_transform(
                np.load("data/audio_features_keptwhisper.npy")
            ),
            "ids": np.load("data/audio_track_ids_keptwhisper.npy"),
        },
        "Dense VAE": {
            "X": np.load("data/audio_latents_vae.npy"),
            "ids": np.load("data/audio_track_ids_keptwhisper.npy"),  # same IDs as MFCC run
        },
        "Conv VAE": {
            "X": np.load("data/audio_latents_convvae.npy"),
            "ids": np.load("data/audio_track_ids_convvae.npy"),
        },
        "Beta-Conv VAE (beta = 1)": {
            "X": np.load("data/audio_latents_betaconvvae_beta1.npy"),
            "ids": np.load("data/audio_track_ids_betaconvvae_beta1.npy"),
        },
        "Beta-Conv VAE (beta = 4)": {
            "X": np.load("data/audio_latents_betaconvvae_beta4.npy"),
            "ids": np.load("data/audio_track_ids_betaconvvae_beta4.npy"),
        },
        "Beta-Conv VAE (beta = 10)": {
            "X": np.load("data/audio_latents_betaconvvae_beta10.npy"),
            "ids": np.load("data/audio_track_ids_betaconvvae_beta10.npy"),
        },
        "Conv Autoencoder": {
            "X": np.load("data/audio_latents_conv_ae.npy"),
            "ids": np.load("data/audio_track_ids_conv_ae.npy"),
        },
    }

    results = []

    for rep_name, pack in reps.items():
        X = pack["X"]
        ids = pack["ids"].astype(int)

        # Align ground-truth genres to this representation's order
        genres_aligned = np.array([tid_to_genre.get(int(t), None) for t in ids], dtype=object)

        # Drop any ids not found in manifest (should be none, but safe)
        ok_mask = genres_aligned != None
        if not np.all(ok_mask):
            X = X[ok_mask]
            ids = ids[ok_mask]
            genres_aligned = genres_aligned[ok_mask]

        print(f"\n--- Representation: {rep_name} | X={X.shape} | ids={len(ids)} ---")

        # ===== KMeans for multiple k =====
        for k in [2, 3, 4, 5]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)

            # Check if clustering produced valid number of clusters
            if is_valid_clustering(labels, min_clusters=2):
                sil = silhouette_score(X, labels)
                dbi = davies_bouldin_score(X, labels)
                ari = adjusted_rand_score(genres_aligned, labels)
                
                fname = safe_filename(rep_name)
                umap_plot(X, labels, f"{rep_name} + KMeans (k={k})", OUT_DIR / f"umap_{fname}_k{k}.png")
                cluster_composition_plot(labels, genres_aligned, f"{rep_name} KMeans (k={k}) composition",
                                         OUT_DIR / f"comp_{fname}_k{k}.png")
            else:
                # Degenerate case: KMeans collapsed to fewer clusters than requested
                n_unique = len(set(labels))
                print(f"  WARNING: KMeans k={k} produced only {n_unique} distinct cluster(s). Setting metrics to NaN.")
                sil = dbi = ari = np.nan

            results.append([rep_name, "KMeans", k, sil, dbi, ari])

        # ===== Agglomerative (fixed k=3 baseline) =====
        agg = AgglomerativeClustering(n_clusters=3)
        labels = agg.fit_predict(X)

        if is_valid_clustering(labels, min_clusters=2):
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            ari = adjusted_rand_score(genres_aligned, labels)
        else:
            n_unique = len(set(labels))
            print(f"  WARNING: Agglomerative produced only {n_unique} distinct cluster(s). Setting metrics to NaN.")
            sil = dbi = ari = np.nan

        results.append([rep_name, "Agglomerative", 3, sil, dbi, ari])

        # ===== DBSCAN =====
        dbs = DBSCAN(eps=0.7, min_samples=10)
        labels = dbs.fit_predict(X)

        # DBSCAN may output noise (-1) and/or a single cluster.
        # For silhouette/DBI, we evaluate only on non-noise points when possible.
        if is_valid_clustering(labels, min_clusters=2):
            # Filter out noise points for metric calculation
            mask_non_noise = labels != -1
            X_nn = X[mask_non_noise]
            y_nn = labels[mask_non_noise]
            g_nn = genres_aligned[mask_non_noise]

            sil = silhouette_score(X_nn, y_nn)
            dbi = davies_bouldin_score(X_nn, y_nn)
            ari = adjusted_rand_score(g_nn, y_nn)
        else:
            n_unique = len(set(labels))
            n_noise = np.sum(labels == -1)
            print(f"  WARNING: DBSCAN produced {n_unique} distinct cluster(s) ({n_noise} noise points). Setting metrics to NaN.")
            sil = dbi = ari = np.nan

        results.append([rep_name, "DBSCAN", "-", sil, dbi, ari])

    df = pd.DataFrame(
        results,
        columns=["representation", "algorithm", "k", "silhouette", "davies_bouldin", "ARI"]
    )

    out_csv = OUT_DIR / "hard_metrics.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== DONE ===")
    print("Metrics saved :", out_csv)
    print("Plots saved   :", OUT_DIR.resolve())
    print("====================================\n")


if __name__ == "__main__":
    main()