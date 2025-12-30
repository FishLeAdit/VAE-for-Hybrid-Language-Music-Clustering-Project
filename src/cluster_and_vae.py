#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

# -------------------------------------------------------------------
# Make imports work when running from src/ directly
# Ensures project root is on sys.path
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.audio_vae import AudioVAE, vae_loss_function
from src.cluster_visualize import cluster_kmeans, plot_tsne


# =============================
# HARD-CODED PATHS
# =============================
X_PATH = "data/audio_features_keptwhisper.npy"
Z_OUT = "data/audio_latents_vae.npy"
PLOT_OUT = "results/tsne_clusters.png"

# =============================
# TRAINING SETTINGS
# =============================
EPOCHS = 150
BATCH_SIZE = 64
LR = 1e-3


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def main():
    X = np.load(X_PATH).astype(np.float32)
    Xn = standardize(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AudioVAE(input_dim=Xn.shape[1]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=LR)

    loader = DataLoader(
        TensorDataset(torch.tensor(Xn, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)
            loss, _, _ = vae_loss_function(recon, xb, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {epoch_loss/len(loader):.4f}")

    # Latents (use mu as deterministic embedding)
    vae.eval()
    latent_vectors = []
    with torch.no_grad():
        for (xb,) in DataLoader(
            TensorDataset(torch.tensor(Xn, dtype=torch.float32)),
            batch_size=256,
            shuffle=False
        ):
            mu, _ = vae.encode(xb.to(device))
            latent_vectors.append(mu.cpu().numpy())

    Z = np.vstack(latent_vectors).astype(np.float32)

    # Save latents
    Path(Z_OUT).parent.mkdir(parents=True, exist_ok=True)
    np.save(Z_OUT, Z)
    print("Saved latents:", Z_OUT, "shape:", Z.shape)

    # Cluster + plot
    labels = cluster_kmeans(Z, k=5)
    plot_tsne(Z, labels, PLOT_OUT)
    print("Saved plot:", PLOT_OUT)

    print("Finished clustering and visualization!")


if __name__ == "__main__":
    main()
