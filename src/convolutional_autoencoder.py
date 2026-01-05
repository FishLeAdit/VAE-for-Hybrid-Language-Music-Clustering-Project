#!/usr/bin/env python3
"""
Conv Autoencoder baseline for log-mel spectrograms.
Same general encoder/decoder style as VAE but:
- No KL term
- Latent is deterministic
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int = 16):
        super().__init__()
        c, h, w = input_shape

        self.enc = nn.Sequential(
            nn.Conv2d(c, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            h_enc = self.enc(dummy)
            self._enc_out_shape = h_enc.shape[1:]
            flat_dim = int(h_enc.numel())

        self.fc_latent = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, c, 4, stride=2, padding=1),
        )

        self._input_shape = (c, h, w)

    def encode(self, x):
        h = self.enc(x)
        h = torch.flatten(h, start_dim=1)
        z = self.fc_latent(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, *self._enc_out_shape)
        x_hat = self.dec(h)
        c, H, W = self._input_shape
        x_hat = x_hat[:, :c, :H, :W]
        return x_hat
    
    @staticmethod
    def _match_shape(x_hat, x_ref):
        _, _, Ht, Wt = x_hat.shape
        _, _, Hr, Wr = x_ref.shape

        if Ht > Hr:
            dh = (Ht - Hr) // 2
            x_hat = x_hat[:, :, dh:dh + Hr, :]
        if Wt > Wr:
            dw = (Wt - Wr) // 2
            x_hat = x_hat[:, :, :, dw:dw + Wr]

        _, _, Ht2, Wt2 = x_hat.shape
        pad_h = max(0, Hr - Ht2)
        pad_w = max(0, Wr - Wt2)
        if pad_h > 0 or pad_w > 0:
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            x_hat = F.pad(x_hat, (pad_left, pad_right, pad_top, pad_bottom))

        return x_hat

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        recon = self._match_shape(recon, x)
        return recon, z


def ae_loss(recon, x):
    return F.mse_loss(recon, x, reduction="mean")
