import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, input_shape=(1, 64, 215), latent_dim=16):
        """
        input_shape: (C, n_mels, n_frames)
        Must match the spectrogram shape coming from spectrogram_utils.
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # /2 again
            nn.ReLU(),
        )

        # Dynamically compute flattened size after convs
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            h = self.encoder(dummy)
            self._enc_shape = h.shape[1:]            # (C, H, W)
            self._flat_dim = h.numel()               # C*H*W

        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.view(x.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, *self._enc_shape)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # IMPORTANT: recon size may be slightly larger due to transpose conv rounding.
        # Crop to match input exactly.
        recon = recon[:, :, :x.shape[2], :x.shape[3]]
        return recon, mu, logvar
