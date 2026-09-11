"""Etapa A: arquitectura del autoencoder de secuencias.

Consume el contrato de ``src.data.sequences`` (``x [B,L,F]``, ``mask [B,L]``)
sin depender de las etiquetas ``y``: el objetivo es aprender una
representación comprimida ``z`` del comportamiento normal de un remitente y
reconstruir la secuencia original desde ``z``. El error de reconstrucción
por secuencia (``masked_reconstruction_error``) será el score de anomalía
una vez entrenado el modelo (entrenamiento, umbral y checkpoint se añaden en
los siguientes commits de esta rama).

``SequenceAutoencoder``
    Encoder Transformer con máscara de padding, pooling por atención hacia
    ``z``, y un decoder que reconstruye las ``max_len`` posiciones a partir
    de ese vector comprimido.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class StageAConfig:
    feature_dim: int
    max_len: int
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


class PositionalEncoding(nn.Module):
    """Codificación posicional senoidal fija, suficiente para max_len<=64."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionPooling(nn.Module):
    """Comprime el encoder en un vector ``z`` con un query aprendido."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, encoded: torch.Tensor, key_padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoded.size(0)
        query = self.query.expand(batch_size, -1, -1)
        pooled, weights = self.attn(
            query, encoded, encoded, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True
        )
        return pooled.squeeze(1), weights.squeeze(1)


class SequenceAutoencoder(nn.Module):
    """Encoder-decoder Transformer que reconstruye secuencias por remitente."""

    def __init__(self, config: StageAConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.feature_dim, config.d_model)
        self.positional = PositionalEncoding(config.d_model, config.max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers)
        self.pooling = AttentionPooling(config.d_model, config.nhead, config.dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, config.num_decoder_layers)
        self.output_proj = nn.Linear(config.d_model, config.feature_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        key_padding_mask = ~mask
        hidden = self.positional(self.input_proj(x))
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        z, attention_weights = self.pooling(encoded, key_padding_mask)
        broadcast = self.positional(z.unsqueeze(1).expand(-1, x.size(1), -1))
        decoded = self.decoder(broadcast)
        x_hat = self.output_proj(decoded)
        return {"x_hat": x_hat, "z": z, "attention_weights": attention_weights}


def masked_reconstruction_error(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Error cuadrático medio por secuencia, ignorando posiciones de padding."""
    squared_error = (x_hat - x).pow(2) * mask.unsqueeze(-1)
    valid_elements = mask.sum(dim=1).clamp(min=1).to(squared_error.dtype) * x.size(-1)
    return squared_error.sum(dim=(1, 2)) / valid_elements
