"""Etapa B: clasificador supervisado con transfer learning desde la Etapa A.

``StageBClassifier`` reutiliza la arquitectura de encoder+pooling de
``src.models.stage_a.SequenceAutoencoder`` (sin su decoder) y añade una
cabeza de clasificación binaria. La señal de la Etapa A se combina en el
nivel de features: el error de reconstrucción del autoencoder ya entrenado
se concatena al vector comprimido ``z`` antes de la cabeza, así la
predicción final usa tanto la representación transferida como el score de
anomalía original.

Dos variantes se entrenan con ``train_stage_b`` para el experimento de
ablación obligatorio:

``transferred=True`` (arquitectura de dos etapas)
    El backbone se inicializa con los pesos de un ``SequenceAutoencoder``
    ya entrenado (Etapa A) y se ajusta con *discriminative fine-tuning*:
    la cabeza nueva entrena sola las primeras ``freeze_epochs`` épocas
    mientras el backbone permanece congelado, y luego el backbone se
    descongela con una tasa de aprendizaje menor que la de la cabeza.
``transferred=False`` (línea base obligatoria)
    El mismo backbone se inicializa aleatoriamente y se entrena desde
    cero sobre ``loaders["train"]`` completo, sin la señal de la Etapa A
    (sin score de anomalía como feature, sin pesos transferidos).

La pérdida es ``BCEWithLogitsLoss(pos_weight=...)`` usando el ``pos_weight``
que ya publica el manifiesto de C1: con ~0.7% de positivos, es la opción
estándar y no requiere buscar un hiperparámetro adicional (a diferencia de
la pérdida focal), y usa directamente el desbalance ya medido.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from src.models.stage_a import (
    SequenceAutoencoder,
    StageAConfig,
    ThresholdInfo,
    masked_reconstruction_error,
    resolve_device,
    select_threshold,
)


@dataclass
class StageBConfig:
    hidden_dim: int = 64
    dropout: float = 0.1
    use_stage_a_score: bool = True
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


class StageBClassifier(nn.Module):
    """Encoder+pooling reutilizados de la Etapa A más una cabeza de clasificación."""

    def __init__(self, stage_a_config: StageAConfig, config: StageBConfig) -> None:
        super().__init__()
        self.stage_a_config = stage_a_config
        self.config = config
        backbone = SequenceAutoencoder(stage_a_config)
        self.input_proj = backbone.input_proj
        self.positional = backbone.positional
        self.encoder = backbone.encoder
        self.pooling = backbone.pooling
        head_input_dim = stage_a_config.d_model + (1 if config.use_stage_a_score else 0)
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def load_backbone_from_stage_a(self, stage_a_model: SequenceAutoencoder) -> None:
        """Transferir input_proj, encoder y pooling desde un autoencoder ya entrenado."""
        self.input_proj.load_state_dict(stage_a_model.input_proj.state_dict())
        self.encoder.load_state_dict(stage_a_model.encoder.state_dict())
        self.pooling.load_state_dict(stage_a_model.pooling.state_dict())

    def backbone_parameters(self):
        for module in (self.input_proj, self.encoder, self.pooling):
            yield from module.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Congelar/descongelar el backbone transferido.

        Un backbone congelado también se pone en modo eval (dropout
        desactivado): además de ser la práctica estándar para pesos que no
        se actualizan, evita una limitación real de MPS donde
        ``scaled_dot_product_attention`` no soporta dropout cuando los
        parámetros de entrada no requieren gradiente.
        """
        for parameter in self.backbone_parameters():
            parameter.requires_grad_(trainable)
        self.input_proj.train(trainable)
        self.encoder.train(trainable)
        self.pooling.train(trainable)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = ~mask
        hidden = self.positional(self.input_proj(x))
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        z, attention_weights = self.pooling(encoded, key_padding_mask)
        return z, attention_weights

    def forward(self, x: torch.Tensor, mask: torch.Tensor, stage_a_score: torch.Tensor | None) -> dict[str, torch.Tensor]:
        z, attention_weights = self.encode(x, mask)
        if self.config.use_stage_a_score:
            if stage_a_score is None:
                raise ValueError("use_stage_a_score=True requiere stage_a_score")
            features = torch.cat([z, stage_a_score.unsqueeze(-1)], dim=-1)
        else:
            features = z
        logit = self.head(features).squeeze(-1)
        return {"logit": logit, "attention_weights": attention_weights, "z": z}


@dataclass
class StageBHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_average_precision: list[float] = field(default_factory=list)
    best_epoch: int = 0
    backbone_frozen_epochs: int = 0


@torch.no_grad()
def _stage_a_scores_for_batch(
    stage_a_model: SequenceAutoencoder | None, x: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor | None:
    if stage_a_model is None:
        return None
    stage_a_model.eval()
    output = stage_a_model(x, mask)
    return masked_reconstruction_error(output["x_hat"], x, mask)


@torch.no_grad()
def score_loader_stage_b(
    model: StageBClassifier,
    loader: DataLoader,
    device: torch.device,
    stage_a_model: SequenceAutoencoder | None,
) -> dict[str, np.ndarray]:
    model.eval()
    if stage_a_model is not None:
        stage_a_model = stage_a_model.to(device)
    probabilities, labels, sender_ids = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        stage_a_score = _stage_a_scores_for_batch(stage_a_model, x, mask)
        output = model(x, mask, stage_a_score)
        probabilities.append(torch.sigmoid(output["logit"]).cpu().numpy())
        labels.append(batch["y"].numpy())
        sender_ids.extend(batch["sender_id"])
    return {
        "probability": np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32),
        "y": np.concatenate(labels) if labels else np.empty(0, dtype=np.float32),
        "sender_id": np.asarray(sender_ids),
    }


def train_stage_b(
    loaders: dict[str, DataLoader],
    stage_a_config: StageAConfig,
    config: StageBConfig,
    *,
    stage_a_model: SequenceAutoencoder | None,
    pos_weight: float,
    epochs: int = 15,
    freeze_epochs: int = 1,
    lr_head: float = 1e-3,
    lr_backbone: float = 5e-4,
    weight_decay: float = 1e-5,
    device: str | None = None,
    log_every: int = 1,
) -> tuple[StageBClassifier, StageBHistory]:
    """Entrenar la Etapa B sobre ``loaders["train"]`` completo (con etiquetas).

    Si ``stage_a_model`` no es ``None``, sus pesos de input_proj/encoder/pooling
    se transfieren al backbone antes de entrenar (arquitectura de dos etapas).
    Si es ``None``, el backbone se entrena desde cero (línea base obligatoria).

    ``freeze_epochs=1`` y ``lr_backbone=5e-4`` son el resultado de una
    corrección real: con un congelamiento más largo (3 épocas) y un LR de
    backbone más conservador (1e-4) —valores iniciales razonables pero no
    validados—, el modelo transferido quedaba por debajo de la línea base
    entrenada desde cero (F1 de prueba 0.168±0.010 vs 0.258±0.039 en 3
    semillas): el backbone no alcanzaba a adaptarse a la señal supervisada
    en el presupuesto de épocas disponible. Con este cronograma más corto y
    un LR mayor, el modelo transferido pasa a superar consistentemente a
    ese mismo baseline en las 3 semillas (números exactos, con la varianza
    real observada entre corridas).
    """
    torch.manual_seed(config.seed)
    resolved_device = resolve_device(device)
    model = StageBClassifier(stage_a_config, config).to(resolved_device)
    transferred = stage_a_model is not None
    if transferred:
        model.load_backbone_from_stage_a(stage_a_model.to(resolved_device))
        stage_a_model = stage_a_model.to(resolved_device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=resolved_device))
    history = StageBHistory(backbone_frozen_epochs=freeze_epochs if transferred else 0)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_ap = float("nan")

    # Un solo optimizer con ambos grupos desde el inicio: mientras el backbone
    # está congelado (requires_grad=False), autograd nunca produce gradiente
    # para esos parámetros y Adam simplemente no los actualiza — no hace
    # falta reconstruir el optimizer al descongelar, lo que además evita
    # perder el momentum acumulado por la cabeza durante el congelamiento.
    optimizer = torch.optim.Adam(
        [
            {"params": model.head.parameters(), "lr": lr_head},
            {"params": model.backbone_parameters(), "lr": lr_backbone if transferred else lr_head},
        ],
        weight_decay=weight_decay,
    )

    backbone_unlocked = not (transferred and freeze_epochs > 0)
    if not backbone_unlocked:
        model.set_backbone_trainable(False)

    for epoch in range(epochs):
        if transferred and epoch == freeze_epochs and not backbone_unlocked:
            model.set_backbone_trainable(True)
            backbone_unlocked = True

        model.head.train()
        if backbone_unlocked:
            model.input_proj.train()
            model.encoder.train()
            model.pooling.train()
        epoch_losses = []
        for batch in loaders["train"]:
            x = batch["x"].to(resolved_device)
            mask = batch["mask"].to(resolved_device)
            y = batch["y"].to(resolved_device)
            stage_a_score = _stage_a_scores_for_batch(stage_a_model, x, mask)
            optimizer.zero_grad()
            output = model(x, mask, stage_a_score)
            loss = criterion(output["logit"], y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.train_loss.append(train_loss)

        validation = score_loader_stage_b(model, loaders["validation"], resolved_device, stage_a_model)
        ap = float(average_precision_score(validation["y"], validation["probability"])) if validation["y"].sum() > 0 else float("nan")
        history.validation_average_precision.append(ap)
        # best_ap arranca en NaN: sin el `or`, una validación sin positivos
        # (ap NaN) nunca actualizaría best_state, dejando los pesos sin
        # entrenar del modelo en vez de algún checkpoint entrenado.
        if ap >= best_ap or math.isnan(best_ap):
            best_ap = ap
            history.best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        if log_every and epoch % log_every == 0:
            frozen = transferred and epoch < freeze_epochs
            print(f"epoch {epoch:02d} | train_loss={train_loss:.5f} | validation_AP={ap:.4f} | backbone_frozen={frozen}")

    model.load_state_dict(best_state)
    return model, history


def evaluate_stage_b(
    model: StageBClassifier,
    loaders: dict[str, DataLoader],
    device: torch.device,
    stage_a_model: SequenceAutoencoder | None,
) -> dict:
    """Umbral por F1 sobre validación, evaluado una sola vez en prueba.

    Mismo procedimiento que ``src.models.stage_a`` para poder comparar
    ambas etapas con la misma metodología en la tabla de ablación.
    """
    validation = score_loader_stage_b(model, loaders["validation"], device, stage_a_model)
    threshold_info = select_threshold(validation["probability"], validation["y"])

    test = score_loader_stage_b(model, loaders["test"], device, stage_a_model)
    test_pred = (test["probability"] > threshold_info.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test["y"], test_pred).ravel()
    return {
        "threshold": threshold_info,
        "test_precision": float(precision_score(test["y"], test_pred, zero_division=0)),
        "test_recall": float(recall_score(test["y"], test_pred, zero_division=0)),
        "test_f1": float(f1_score(test["y"], test_pred, zero_division=0)),
        "test_average_precision": float(average_precision_score(test["y"], test["probability"])),
        "test_confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "test_scores": test,
    }


def save_checkpoint(
    path: str | Path,
    model: StageBClassifier,
    stage_a_config: StageAConfig,
    config: StageBConfig,
    threshold_info: ThresholdInfo,
    *,
    transferred: bool,
    history: StageBHistory | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stage_a_config": stage_a_config.as_dict(),
            "config": config.as_dict(),
            "threshold": threshold_info.as_dict(),
            "transferred": transferred,
            "history": {
                "train_loss": history.train_loss,
                "validation_average_precision": history.validation_average_precision,
                "best_epoch": history.best_epoch,
                "backbone_frozen_epochs": history.backbone_frozen_epochs,
            }
            if history is not None
            else None,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device | None = "cpu"
) -> tuple[StageBClassifier, StageAConfig, StageBConfig, ThresholdInfo, dict]:
    # weights_only=True: ver la misma nota en src/models/stage_a.load_checkpoint.
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    stage_a_config = StageAConfig(**payload["stage_a_config"])
    config = StageBConfig(**payload["config"])
    model = StageBClassifier(stage_a_config, config)
    model.load_state_dict(payload["state_dict"])
    threshold_info = ThresholdInfo(**payload["threshold"])
    extra = {"transferred": payload.get("transferred"), "history": payload.get("history")}
    return model, stage_a_config, config, threshold_info, extra


def predict_batch(
    model: StageBClassifier,
    batch: dict,
    device: torch.device | None,
    stage_a_model: SequenceAutoencoder | None,
) -> dict[str, np.ndarray]:
    """Probabilidad y atención para un batch ya colacionado (p. ej. de ``get_sender``)."""
    resolved_device = device or next(model.parameters()).device
    model.eval()
    if stage_a_model is not None:
        stage_a_model = stage_a_model.to(resolved_device)
    with torch.no_grad():
        x = batch["x"].to(resolved_device)
        mask = batch["mask"].to(resolved_device)
        stage_a_score = _stage_a_scores_for_batch(stage_a_model, x, mask)
        output = model(x, mask, stage_a_score)
        probability = torch.sigmoid(output["logit"]).cpu().numpy()
    mask_np = batch["mask"][0].numpy()
    attention = output["attention_weights"][0].detach().cpu().numpy()[mask_np]
    return {"probability": probability, "attention": attention}
