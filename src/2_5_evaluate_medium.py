#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

import matplotlib.pyplot as plt

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ------------ inputs ------------
MANIFEST = "data/manifest_english_only.csv"

# baseline audio features (MFCC)
MFCC_X = "data/audio_features_keptwhisper.npy"
MFCC_IDS = "data/audio_track_ids_keptwhisper.npy"  # if you don't have this, set to None and align by manifest only

# dense VAE latents
DENSE_Z = "data/audio_latents_vae.npy"

# conv VAE latents
CONV_Z = "data/audio_latents_convvae.npy"
CONV_IDS = "data/audio_track_ids_convvae.npy"

# fused
FUSED_X = "data/fused_audio_lyrics.npy"
FUSED_IDS = "data/fused_track_ids.npy"

# outputs
OUT_DIR = Path("results/medium")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_METRICS = OUT_DIR / "medium_metrics.csv"


# ------------ helpers ------------
def safe_metrics(X, labels, y_true=None):
    # Handle cases where clustering degenerates (e.g., DBSCAN all noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "ari": None if y_true is None else None,
            "n_clusters": n_clusters,
            "noise": int((labels == -1).sum()) if -1 in labels else 0
        }

    out = {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "n_clusters": n_clusters,
        "noise": int((labels == -1).sum()) if -1 in labels else 0
    }
    if y_true is not None:
        out["ari"] = float(adjusted_rand_score(y_true, labels))
    else:
        out["ari"] = None
    return out


def umap_plot(X, labels, title, outpath):
    if not HAS_UMAP:
        return
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(X)
    plt.figure(figsize=(6,5))
    plt.scatter(emb[:,0], emb[:,1], c=labels, s=6, cmap="tab10")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def align_by_ids(repr_X, repr_ids, target_ids):
    id_to_row = {int(tid): repr_X[i] for i, tid in enumerate(repr_ids)}
    rows = []
    kept = []
    for tid in target_ids:
        if int(tid) in id_to_row:
            rows.append(id_to_row[int(tid)])
            kept.append(int(tid))
    return np.vstack(rows).astype(np.float32), np.array(kept, dtype=np.int32)


def main():
    df = pd.read_csv(MANIFEST)
    df["track_id"] = df["track_id"].astype(int)

    # Partial labels (genre) for ARI
    genres = df["genre"].astype(str).to_numpy()
    genre_codes = pd.factorize(genres)[0]

    target_ids = df["track_id"].to_numpy()

    results = []

    # Representation A: MFCC -> PCA(8) 
    X_mfcc = np.load(MFCC_X)
    if os.path.exists(MFCC_IDS):
        mfcc_ids = np.load(MFCC_IDS).astype(int)
        X_mfcc, aligned_ids = align_by_ids(X_mfcc, mfcc_ids, target_ids)
        # align labels too
        mask = df["track_id"].isin(aligned_ids)
        y_true = pd.factorize(df.loc[mask, "genre"].astype(str))[0]
    else:
        # fallback: assume MFCC already in target order (not ideal)
        y_true = genre_codes

    X_mfcc = StandardScaler().fit_transform(X_mfcc).astype(np.float32)
    X_pca = PCA(n_components=8, random_state=42).fit_transform(X_mfcc).astype(np.float32)

    #  Representation B: Dense VAE latents 
    Z_dense = np.load(DENSE_Z).astype(np.float32)
    Z_dense = Z_dense[:len(y_true)]
    Z_dense = StandardScaler().fit_transform(Z_dense).astype(np.float32)

    #  Representation C: Conv VAE latents 
    Z_conv = np.load(CONV_Z).astype(np.float32)
    conv_ids = np.load(CONV_IDS).astype(int)
    Z_conv, conv_aligned = align_by_ids(Z_conv, conv_ids, target_ids)
    mask = df["track_id"].isin(conv_aligned)
    y_conv = pd.factorize(df.loc[mask, "genre"].astype(str))[0]
    Z_conv = StandardScaler().fit_transform(Z_conv).astype(np.float32)

    #  Representation D: Fused audio+lyrics 
    X_fused = np.load(FUSED_X).astype(np.float32)
    fused_ids = np.load(FUSED_IDS).astype(int)
    mask = df["track_id"].isin(fused_ids)
    y_fused = pd.factorize(df.loc[mask, "genre"].astype(str))[0]
    X_fused = StandardScaler().fit_transform(X_fused).astype(np.float32)

    reps = [
        ("PCA(8) on MFCC", X_pca, y_true),
        ("DenseVAE latent", Z_dense, y_true),
        ("ConvVAE latent", Z_conv, y_conv),
        ("Fused(ConvVAE+Lyrics)", X_fused, y_fused),
    ]

    # clustering configs
    k_list = [2,3,4,5,6]

    for rep_name, X, y in reps:
        # KMeans sweep
        for k in k_list:
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X)
            m = safe_metrics(X, labels, y_true=y)
            results.append({
                "representation": rep_name,
                "clusterer": f"KMeans(k={k})",
                **m
            })
            umap_plot(X, labels, f"{rep_name} + KMeans(k={k})", OUT_DIR / f"umap_{rep_name.replace(' ','_')}_kmeans_{k}.png")

        # Agglomerative sweep
        for k in k_list:
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            m = safe_metrics(X, labels, y_true=y)
            results.append({
                "representation": rep_name,
                "clusterer": f"Agglomerative(k={k})",
                **m
            })

        # DBSCAN (few sensible defaults; tune later)
        for eps in [0.5, 0.8, 1.2]:
            labels = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
            m = safe_metrics(X, labels, y_true=y)
            results.append({
                "representation": rep_name,
                "clusterer": f"DBSCAN(eps={eps},min_samples=10)",
                **m
            })

    out = pd.DataFrame(results)
    out.to_csv(OUT_METRICS, index=False)

    print("Saved metrics:", OUT_METRICS)
    print(out.sort_values(["silhouette"], ascending=False).head(15))

    # quick auto-discussion hints
    best = out.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).head(5)
    print("\nTop 5 by Silhouette:")
    print(best[["representation","clusterer","silhouette","davies_bouldin","calinski_harabasz","ari","n_clusters","noise"]])

if __name__ == "__main__":
    main()
