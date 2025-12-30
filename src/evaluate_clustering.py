#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE

# Optional UMAP
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# =============================
# HARD-CODED INPUT PATHS
# =============================
X_PATH = "data/audio_features_keptwhisper.npy"   # MFCC features (N, 40)
Z_PATH = "data/audio_latents_vae.npy"            # VAE latent means (N, latent_dim)  <-- you may need to save this
IDS_PATH = "data/audio_track_ids_keptwhisper.npy"

# =============================
# OUTPUTS
# =============================
OUT_DIR = Path("results")
PLOTS_DIR = OUT_DIR / "plots"
METRICS_CSV = OUT_DIR / "cluster_metrics.csv"

# =============================
# SETTINGS
# =============================
K_LIST = [2, 3, 4, 5, 6, 8, 10]
PCA_COMPONENTS = 8   # baseline dimensionality for PCA
TSNE_PERPLEXITY = 30
RANDOM_STATE = 42


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def run_kmeans_and_metrics(X: np.ndarray, k: int):
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return labels, sil, ch


def plot_2d(points_2d: np.ndarray, labels: np.ndarray, title: str, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, s=6)
    plt.title(title)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()

    X = np.load(X_PATH).astype(np.float32)
    X = standardize(X)

    # Latents: if missing, you must generate/save from your VAE code
    Z = np.load(Z_PATH).astype(np.float32)

    # Baseline features (PCA on MFCC)
    X_pca = PCA(n_components=min(PCA_COMPONENTS, X.shape[1]), random_state=RANDOM_STATE).fit_transform(X)

    rows = []

    for k in K_LIST:
        # ===== VAE latent clustering =====
        labels_vae, sil_vae, ch_vae = run_kmeans_and_metrics(Z, k)
        rows.append({
            "method": "VAE_latent + KMeans",
            "k": k,
            "silhouette": sil_vae,
            "calinski_harabasz": ch_vae
        })

        # t-SNE plot for VAE (only for a couple k's if you want less files)
        tsne_vae = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init="pca",
                        learning_rate="auto", random_state=RANDOM_STATE).fit_transform(Z)
        plot_2d(
            tsne_vae, labels_vae,
            title=f"t-SNE of VAE latents (k={k})",
            outpath=PLOTS_DIR / f"tsne_vae_k{k}.png"
        )

        # Optional UMAP plot for VAE
        if HAS_UMAP:
            um = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
            umap_vae = um.fit_transform(Z)
            plot_2d(
                umap_vae, labels_vae,
                title=f"UMAP of VAE latents (k={k})",
                outpath=PLOTS_DIR / f"umap_vae_k{k}.png"
            )

        # ===== PCA baseline clustering =====
        labels_pca, sil_pca, ch_pca = run_kmeans_and_metrics(X_pca, k)
        rows.append({
            "method": f"PCA({X_pca.shape[1]}) + KMeans",
            "k": k,
            "silhouette": sil_pca,
            "calinski_harabasz": ch_pca
        })

        # t-SNE plot for PCA baseline
        tsne_pca = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init="pca",
                        learning_rate="auto", random_state=RANDOM_STATE).fit_transform(X_pca)
        plot_2d(
            tsne_pca, labels_pca,
            title=f"t-SNE of PCA features (k={k})",
            outpath=PLOTS_DIR / f"tsne_pca_k{k}.png"
        )

        # Optional UMAP plot for PCA baseline
        if HAS_UMAP:
            um = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
            umap_pca = um.fit_transform(X_pca)
            plot_2d(
                umap_pca, labels_pca,
                title=f"UMAP of PCA features (k={k})",
                outpath=PLOTS_DIR / f"umap_pca_k{k}.png"
            )

        print(f"Done k={k} | VAE sil={sil_vae:.4f} CH={ch_vae:.1f} | PCA sil={sil_pca:.4f} CH={ch_pca:.1f}")

    df = pd.DataFrame(rows).sort_values(["k", "method"])
    df.to_csv(METRICS_CSV, index=False)
    print("\nSaved metrics:", METRICS_CSV)
    print("Saved plots to:", PLOTS_DIR.resolve())
    print("\nTop rows:\n", df.head(10))


if __name__ == "__main__":
    main()
