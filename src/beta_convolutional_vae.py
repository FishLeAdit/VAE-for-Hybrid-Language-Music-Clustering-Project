#!/usr/bin/env python3
"""
Beta-ConvVAE: Conv VAE for (1, mel, frames) log-mel spectrograms.

- Learns latent z ~ N(mu, sigma)
- Loss = recon + beta * KL
- Encoder/decoder are convolutional; FC layers are shaped dynamically from input_shape.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaConvVAE(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int = 16):
        """
        input_shape: (C, H, W) e.g. (1, 64, 215)
        """
        super().__init__()
        c, h, w = input_shape

        # Encoder conv stack
        self.enc = nn.Sequential(
            nn.Conv2d(c, 32, 4, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),# /8
            nn.ReLU(inplace=True),
        )

        # Infer flattened size dynamically (no hardcoding)
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            h_enc = self.enc(dummy)
            self._enc_out_shape = h_enc.shape[1:]  # (C', H', W')
            flat_dim = int(h_enc.numel())

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: map latent -> conv feature map -> deconvs
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        c2, h2, w2 = self._enc_out_shape
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
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
            h = self.fc_dec(z)
            h = h.view(-1, *self._enc_out_shape)  # (B, C', H', W')
            x_hat = self.dec(h)
            return x_hat

    @staticmethod
    def _match_shape(x_hat, x_ref):
        """
        Make x_hat match x_ref by center-cropping or zero-padding on H/W.
        """
        _, _, Ht, Wt = x_hat.shape
        _, _, Hr, Wr = x_ref.shape

        # Crop if too large
        if Ht > Hr:
            dh = (Ht - Hr) // 2
            x_hat = x_hat[:, :, dh:dh + Hr, :]
        if Wt > Wr:
            dw = (Wt - Wr) // 2
            x_hat = x_hat[:, :, :, dw:dw + Wr]

        # Pad if too small
        _, _, Ht2, Wt2 = x_hat.shape
        pad_h = max(0, Hr - Ht2)
        pad_w = max(0, Wr - Wt2)

        if pad_h > 0 or pad_w > 0:
            # pad format: (left, right, top, bottom)
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            x_hat = F.pad(x_hat, (pad_left, pad_right, pad_top, pad_bottom))

        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # CRITICAL: enforce exact shape match (fixes 208 vs 215)
        recon = self._match_shape(recon, x)

        return recon, mu, logvar

def beta_vae_loss(recon, x, mu, logvar, beta: float = 4.0):
    # MSE recon in log-mel space
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl
    return loss, recon_loss, kl
