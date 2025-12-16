import torch
import torch.nn as nn


class TextDecoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, latent_dim, num_layers=2, nhead=4, max_len=50
    ):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # Project latent to embedding dimension if needed
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, latent, target_seq=None):
        # latent: (B, latent_dim)
        # target_seq: (B, seq_len) - for training with teacher forcing

        batch_size = latent.size(0)

        # Use latent as memory for the transformer decoder?
        # Or initialize the state?
        # Standard way: Use latent as the "memory" (encoder output)

        memory = self.latent_proj(latent).unsqueeze(1)  # (B, 1, embed_dim)

        if target_seq is not None:
            # Training mode
            tgt_emb = (
                self.embedding(target_seq)
                + self.pos_encoder[:, : target_seq.size(1), :]
            )
            output = self.transformer_decoder(tgt_emb, memory)
            return self.fc_out(output)
        else:
            # Inference mode (greedy search for simplicity)
            # This is a placeholder for actual generation logic
            return None


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels=3):
        super().__init__()
        # Simple CNN Decoder
        # Latent -> 512 -> Reshape (512, 1, 1) -> Deconvs -> (3, 224, 224)
        # This is a very simplified decoder.

        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, output_channels, kernel_size=4, stride=2, padding=1
            ),  # 224x224
            nn.Sigmoid(),  # Assuming normalized images [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 7, 7)
        x = self.decoder(x)
        return x
