#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.beta_convolutional_vae import BetaConvVAE, beta_vae_loss

X_PATH = "data/logmel_X.npy"
IDS_PATH = "data/logmel_ids.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
EPOCHS = 30
LATENT = 16
LR = 1e-3

BETAS = [1.0, 4.0, 10.0]  # you can change

def main():
    X = np.load(X_PATH).astype(np.float32)
    ids = np.load(IDS_PATH).astype(np.int32)
    X_t = torch.tensor(X, dtype=torch.float32)

    # normalize per-dataset (stabilizes training)
    mean = X_t.mean()
    std = X_t.std().clamp_min(1e-6)
    X_t = (X_t - mean) / std

    loader = DataLoader(TensorDataset(X_t), batch_size=BATCH, shuffle=True)

    input_shape = tuple(X_t.shape[1:])  # (1,64,215)

    for beta in BETAS:
        model = BetaConvVAE(input_shape=input_shape, latent_dim=LATENT).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        print(f"\nTraining BetaConvVAE | beta={beta} | device={DEVICE} | N={len(X_t)}")
        for ep in range(1, EPOCHS + 1):
            model.train()
            total = 0.0
            pbar = tqdm(loader, desc=f"beta={beta} ep={ep}/{EPOCHS}", unit="batch", leave=False)
            for (xb,) in pbar:
                xb = xb.to(DEVICE)
                recon, mu, logvar = model(xb)
                loss, r, kl = beta_vae_loss(recon, xb, mu, logvar, beta=beta)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

                total += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{r.item():.4f}", kl=f"{kl.item():.4f}")

            if ep % 5 == 0 or ep == 1:
                print(f"beta={beta} epoch={ep:>2} avg_loss={total/len(loader):.4f}")

        # Encode mu
        model.eval()
        mus = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(X_t), batch_size=128, shuffle=False):
                mu, _ = model.encode(xb.to(DEVICE))
                mus.append(mu.cpu().numpy())
        Z = np.vstack(mus).astype(np.float32)

        out_z = f"data/audio_latents_betaconvvae_beta{beta:g}.npy"
        out_ids = f"data/audio_track_ids_betaconvvae_beta{beta:g}.npy"
        np.save(out_z, Z)
        np.save(out_ids, ids)

        print(f"Saved latents: {out_z} {Z.shape}")
        print(f"Saved ids   : {out_ids} {ids.shape}")

if __name__ == "__main__":
    main()
