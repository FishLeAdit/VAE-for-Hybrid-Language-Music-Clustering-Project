from __future__ import annotations

import os
import sys
import math
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------- project import setup ----------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.convolutional_vae_spectrogram import ConvVAE
from src.spectrogram_utils import audio_to_logmel


# ---------- hardcoded paths ----------
MANIFEST = "data/fma_manifest_5k_5genres_lyrics_whisper_dropped_removed.csv"
AUDIO_ROOT = Path("data/fma_small/fma_small")
OUT_Z = "data/audio_latents_convvae.npy"
OUT_IDS = "data/audio_track_ids_convvae.npy"

# ---------- training config ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
LATENT_DIM = 16
LR = 1e-3

# stability knobs
LOGVAR_CLAMP = (-10.0, 10.0)   # prevents exp overflow in KL
GRAD_CLIP = 5.0                # optional: avoid exploding grads


def tid_to_mp3(tid: int) -> Path:
    fname = f"{tid:06d}.mp3"
    return AUDIO_ROOT / fname[:3] / fname


@contextmanager
def mute_stderr():
    """Hide mpg123 / audioread stderr spam during decoding."""
    old = sys.stderr
    try:
        with open(os.devnull, "w") as f:
            sys.stderr = f
            yield
    finally:
        sys.stderr = old


def load_specs(track_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X_np: (N, 1, 64, T)
      ids_np: (N,)
    Skips undecodable audio.
    """
    specs: list[np.ndarray] = []
    kept: list[int] = []

    missing = 0
    failed = 0

    pbar = tqdm(track_ids, desc="log-mel extraction", unit="track")
    for tid in pbar:
        path = tid_to_mp3(tid)
        if not path.exists():
            missing += 1
            continue

        try:
            with mute_stderr():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    S = audio_to_logmel(str(path))

            # enforce finite in case spectrogram_utils didn't already
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            specs.append(S)
            kept.append(tid)

        except Exception:
            failed += 1
            continue

        if (len(kept) + failed + missing) % 500 == 0:
            pbar.write(f"[status] loaded={len(kept)} failed={failed} missing={missing}")

    if not specs:
        raise RuntimeError("No spectrograms loaded. Check decoding and paths.")

    X_np = np.stack(specs, axis=0)[:, None, :, :]  # (N, 1, mel, frames)
    ids_np = np.asarray(kept, dtype=np.int32)

    print("\n--- load summary ---")
    print("manifest rows :", len(track_ids))
    print("loaded        :", len(kept))
    print("failed decode :", failed)
    print("missing files :", missing)
    print("X_np shape    :", X_np.shape)
    print("--------------------\n")

    return X_np, ids_np


def standardize_tensor(X: torch.Tensor) -> torch.Tensor:
    """
    Global standardization for stability:
      X <- (X - mean) / (std + eps)
    plus hard finite enforcement.
    """
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-6)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def vae_losses(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple VAE loss:
      recon MSE + KL
    with logvar clamp to avoid overflow.
    """
    logvar = torch.clamp(logvar, min=LOGVAR_CLAMP[0], max=LOGVAR_CLAMP[1])

    recon_loss = torch.nn.functional.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl
    return loss, recon_loss, kl


def train_and_encode(X: torch.Tensor) -> np.ndarray:
    """
    Train ConvVAE and return latent means Z (N, latent_dim).
    """
    input_shape = (X.shape[1], X.shape[2], X.shape[3])  # (C, mel, frames)
    model = ConvVAE(input_shape=input_shape, latent_dim=LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)

    print(f"Device: {DEVICE}")
    print("Training ConvVAE...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        loop = tqdm(loader, desc=f"epoch {epoch}/{EPOCHS}", unit="batch", leave=False)
        for (xb,) in loop:
            xb = xb.to(DEVICE)

            recon, mu, logvar = model(xb)
            loss, rloss, kl = vae_losses(recon, xb, mu, logvar)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected. recon={rloss.item()} kl={kl.item()}"
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()

            total += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", recon=f"{rloss.item():.4f}", kl=f"{kl.item():.4f}")

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:>3}/{EPOCHS} | avg_loss={total/len(loader):.4f}")

    print("\nEncoding latents (mu)...")
    model.eval()

    zs = []
    with torch.no_grad():
        enc_loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)
        for (xb,) in tqdm(enc_loader, desc="encode", unit="batch"):
            xb = xb.to(DEVICE)
            mu, _ = model.encode(xb)
            zs.append(mu.detach().cpu().numpy())

    Z = np.vstack(zs).astype(np.float32)
    return Z


def main():
    df = pd.read_csv(MANIFEST)
    if "track_id" not in df.columns:
        raise ValueError("Expected 'track_id' column in manifest.")

    track_ids = df["track_id"].astype(int).tolist()

    #1 load spectrograms
    X_np, ids_np = load_specs(track_ids)

    #2 to tensor + normalize
    X = torch.tensor(X_np, dtype=torch.float32)
    print("Pre-normalize finite?:", torch.isfinite(X).all().item(), "| min/max:", X.min().item(), X.max().item())

    X = standardize_tensor(X)
    print("Post-normalize finite?:", torch.isfinite(X).all().item(), "| min/max:", X.min().item(), X.max().item())

    #3 train + encode
    Z = train_and_encode(X)

    #SAVE
    np.save(OUT_Z, Z)
    np.save(OUT_IDS, ids_np)

    print("\n====== DONE ======")
    print("Latents saved :", OUT_Z, Z.shape)
    print("IDs saved     :", OUT_IDS, ids_np.shape)
    print("==================\n")


if __name__ == "__main__":
    main()
