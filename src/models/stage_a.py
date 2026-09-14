"""Etapa A: autoencoder de secuencias para aprender comportamiento normal.

Consume el contrato de ``src.data.sequences`` (``x [B,L,F]``, ``mask [B,L]``)
y no depende de las etiquetas ``y`` para actualizar sus pesos:
``train_stage_a`` solo itera ``loaders["train_normal"]``. El error de
reconstrucción por secuencia (``masked_reconstruction_error``) es el score
de anomalía; el umbral se elige después, en validación (siguiente commit).

``SequenceAutoencoder``
    Encoder Transformer con máscara de padding, pooling por atención hacia
    un vector comprimido ``z``, y un decoder que reconstruye las ``max_len``
    posiciones a partir de ese vector.
``train_stage_a``
    Bucle de entrenamiento sobre normalidad con selección de checkpoint por
    AUC-PR de validación (métrica, no gradiente).
``select_threshold``
    Barrido sobre la curva precisión-recall de validación; elige el umbral
    que maximiza F1, justificado por la prevalencia extrema (~0.7% positivos).
``save_checkpoint`` / ``load_checkpoint``
    Persisten pesos, configuración y métricas para que el MVP y la Etapa B
    reconstruyan el mismo modelo sin reajustar hiperparámetros.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader


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
        # enable_nested_tensor=False: la ruta rápida de nested tensors para
        # padding mask no está implementada en MPS; desactivarla no cambia
        # el resultado, solo evita ese fast-path interno de PyTorch.
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_encoder_layers, enable_nested_tensor=False)
        self.pooling = AttentionPooling(config.d_model, config.nhead, config.dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, config.num_decoder_layers, enable_nested_tensor=False)
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


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def score_loader(model: SequenceAutoencoder, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    scores, labels, sender_ids, entity_ids = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        output = model(x, mask)
        error = masked_reconstruction_error(output["x_hat"], x, mask)
        scores.append(error.cpu().numpy())
        labels.append(batch["y"].numpy())
        sender_ids.extend(batch["sender_id"])
        entity_ids.extend(batch["entity_id"])
    return {
        "score": np.concatenate(scores) if scores else np.empty(0, dtype=np.float32),
        "y": np.concatenate(labels) if labels else np.empty(0, dtype=np.float32),
        "sender_id": np.asarray(sender_ids),
        "entity_id": np.asarray(entity_ids),
    }


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_average_precision: list[float] = field(default_factory=list)
    best_epoch: int = 0


def train_stage_a(
    loaders: dict[str, DataLoader],
    config: StageAConfig,
    *,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str | None = None,
    log_every: int = 1,
) -> tuple[SequenceAutoencoder, TrainingHistory]:
    """Entrenar exclusivamente sobre ``loaders["train_normal"]``.

    La AUC-PR de validación se calcula cada época solo para elegir el mejor
    checkpoint (no participa en el gradiente), porque con ~0.7% de positivos
    en validación es una señal de separabilidad más informativa que la
    pérdida de reconstrucción por sí sola.
    """
    torch.manual_seed(config.seed)
    resolved_device = resolve_device(device)
    model = SequenceAutoencoder(config).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = TrainingHistory()
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_ap = -1.0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch in loaders["train_normal"]:
            x = batch["x"].to(resolved_device)
            mask = batch["mask"].to(resolved_device)
            optimizer.zero_grad()
            output = model(x, mask)
            per_sequence_error = masked_reconstruction_error(output["x_hat"], x, mask)
            loss = per_sequence_error.mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        validation = score_loader(model, loaders["validation"], resolved_device)
        if validation["y"].sum() > 0:
            ap = float(average_precision_score(validation["y"], validation["score"]))
        else:
            ap = float("nan")
        history.validation_average_precision.append(ap)
        if ap >= best_ap:
            best_ap = ap
            history.best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:02d} | train_loss={train_loss:.5f} | validation_AP={ap:.4f}")

    model.load_state_dict(best_state)
    return model, history


@dataclass
class ThresholdInfo:
    threshold: float
    average_precision: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return asdict(self)


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> ThresholdInfo:
    """Elegir el umbral que maximiza F1 sobre la curva precisión-recall.

    Con clases tan desbalanceadas (~0.7% positivos en validación), exactitud
    o un umbral fijo por percentil no son justificables por sí mismos; la
    curva precisión-recall resume el compromiso real y F1 es la métrica que
    el equipo reporta como principal para este problema (ver justificación
    en el reporte de la Etapa A).
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if labels.sum() == 0:
        raise ValueError("select_threshold requiere al menos un positivo en validación")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_index = int(np.argmax(f1[:-1])) if len(thresholds) else int(np.argmax(f1))
    threshold = float(thresholds[best_index]) if len(thresholds) else float(scores.max())
    average_precision = float(average_precision_score(labels, scores))
    return ThresholdInfo(
        threshold=threshold,
        average_precision=average_precision,
        precision=float(precision[best_index]),
        recall=float(recall[best_index]),
        f1=float(f1[best_index]),
    )


def anomaly_score(model: SequenceAutoencoder, batch: dict, device: torch.device | None = None) -> np.ndarray:
    """Score de anomalía para un batch ya colacionado (p. ej. de ``get_sender``)."""
    resolved_device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = batch["x"].to(resolved_device)
        mask = batch["mask"].to(resolved_device)
        output = model(x, mask)
        error = masked_reconstruction_error(output["x_hat"], x, mask)
    return error.cpu().numpy()


def save_checkpoint(
    path: str | Path,
    model: SequenceAutoencoder,
    config: StageAConfig,
    threshold_info: ThresholdInfo,
    *,
    feature_names: Iterable[str] | None = None,
    history: TrainingHistory | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config.as_dict(),
        "threshold": threshold_info.as_dict(),
        "feature_names": list(feature_names) if feature_names is not None else None,
        "history": {
            "train_loss": history.train_loss,
            "validation_average_precision": history.validation_average_precision,
            "best_epoch": history.best_epoch,
        }
        if history is not None
        else None,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device | None = "cpu"
) -> tuple[SequenceAutoencoder, StageAConfig, ThresholdInfo, dict]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = StageAConfig(**payload["config"])
    model = SequenceAutoencoder(config)
    model.load_state_dict(payload["state_dict"])
    threshold_info = ThresholdInfo(**payload["threshold"])
    extra = {"feature_names": payload.get("feature_names"), "history": payload.get("history")}
    return model, config, threshold_info, extra
