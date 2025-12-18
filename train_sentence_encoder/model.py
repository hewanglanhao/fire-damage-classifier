from typing import Dict

import torch
import torch.nn as nn


class MiniLMEncoder(nn.Module):
    """
    A lightweight Transformer encoder inspired by all-MiniLM-L6.
    - 6 encoder layers
    - Hidden size defaults to 512
    - Uses mean pooling over non-pad tokens for the sentence embedding
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pool_method: str = "mean",
    ):
        super().__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.pool_method = pool_method

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.embed_layer_norm = nn.LayerNorm(hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlm_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Embeddings
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embed_dropout(self.embed_layer_norm(x))

        # Transformer
        padding_mask = attention_mask == 0  # True where padding
        sequence_output = self.encoder(x, src_key_padding_mask=padding_mask)

        # Pool sentence embedding
        if self.pool_method == "cls":
            sentence_embedding = sequence_output[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1)
            summed = (sequence_output * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            sentence_embedding = summed / denom

        # MLM logits
        mlm_logits = self.mlm_head(sequence_output)
        return {
            "sequence_output": sequence_output,
            "sentence_embedding": sentence_embedding,
            "mlm_logits": mlm_logits,
        }
