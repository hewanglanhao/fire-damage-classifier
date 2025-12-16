import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

        # Assuming encoder outputs a feature vector, we project it to mu and logvar
        # This might need adjustment based on encoder output size
        # For now, we assume encoder handles the projection to 2 * latent_dim or we add it here.
        # Let's add it here for flexibility.
        self.fc_mu = None
        self.fc_logvar = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_loss=False):
        # This is a generic forward.
        # Specific implementations might override this or usage might differ.
        pass


class VAEOutput:
    def __init__(self, reconstructed, mu, logvar, latent):
        self.reconstructed = reconstructed
        self.mu = mu
        self.logvar = logvar
        self.latent = latent
