"""Integración de inferencia para el MVP: Etapa A y Etapa B.

``run_stage_a`` carga un checkpoint de ``src.models.stage_a`` y produce el
score de anomalía, el veredicto contra el umbral y los pesos de atención del
pooling (alineados con las transacciones de ``get_sender``, en el mismo
orden). ``run_stage_b`` hace lo mismo con un checkpoint de
``src.models.stage_b``; si no se le pasa un checkpoint (o el archivo no
existe todavía) devuelve ``None`` y el MVP sigue funcionando solo con la
Etapa A en vez de simular un resultado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.stage_a import SequenceAutoencoder, StageAConfig, ThresholdInfo, load_checkpoint, masked_reconstruction_error
from src.models.stage_b import StageBClassifier
from src.models.stage_b import load_checkpoint as load_stage_b_checkpoint
from src.models.stage_b import predict_batch as predict_stage_b_batch


@dataclass
class StageAResult:
    score: float
    threshold: float
    is_anomalous: bool
    average_precision: float
    attention: np.ndarray


@dataclass
class StageBResult:
    probability: float
    threshold: float
    is_anomalous: bool
    contributions: np.ndarray


def load_stage_a(checkpoint_path: str | Path) -> tuple[SequenceAutoencoder, StageAConfig, ThresholdInfo, dict]:
    return load_checkpoint(checkpoint_path)


def run_stage_a(model: SequenceAutoencoder, threshold_info: ThresholdInfo, batch: dict) -> StageAResult:
    """Score, veredicto y atención para un batch de tamaño 1 de ``get_sender``."""
    device = next(model.parameters()).device
    model.eval()
    x = batch["x"].to(device)
    mask_tensor = batch["mask"].to(device)
    with torch.no_grad():
        output = model(x, mask_tensor)
        score = float(masked_reconstruction_error(output["x_hat"], x, mask_tensor)[0])
    mask = batch["mask"][0].numpy()
    attention = output["attention_weights"][0].detach().cpu().numpy()[mask]
    return StageAResult(
        score=score,
        threshold=threshold_info.threshold,
        is_anomalous=score > threshold_info.threshold,
        average_precision=threshold_info.average_precision,
        attention=attention,
    )


def load_stage_b(checkpoint_path: str | Path) -> tuple[StageBClassifier, StageAConfig, ThresholdInfo, dict] | None:
    """Cargar el checkpoint de la Etapa B, o ``None`` si todavía no existe."""
    path = Path(checkpoint_path)
    if not path.is_file():
        return None
    model, stage_a_config, _config, threshold_info, extra = load_stage_b_checkpoint(path)
    return model, stage_a_config, threshold_info, extra


def run_stage_b(
    model: StageBClassifier | None,
    stage_a_model: SequenceAutoencoder | None,
    threshold_info: ThresholdInfo | None,
    batch: dict,
) -> StageBResult | None:
    """Probabilidad y contribuciones por transacción para un batch de ``get_sender``.

    Devuelve ``None`` si no hay checkpoint de la Etapa B todavía, para que el
    MVP siga funcionando solo con la Etapa A en vez de simular un resultado.
    """
    if model is None or threshold_info is None:
        return None
    device = next(model.parameters()).device
    prediction = predict_stage_b_batch(model, batch, device, stage_a_model)
    probability = float(prediction["probability"][0])
    return StageBResult(
        probability=probability,
        threshold=threshold_info.threshold,
        is_anomalous=probability > threshold_info.threshold,
        contributions=prediction["attention"],
    )


def top_contributions(attention: np.ndarray, transactions: list[dict], top_k: int = 3) -> list[dict]:
    order = np.argsort(attention)[::-1][:top_k]
    return [
        {
            "posición": int(index),
            "peso_atención": float(attention[index]),
            "timestamp": transactions[index]["timestamp"],
            "monto_pagado": transactions[index]["amount_paid"],
            "moneda_pago": transactions[index]["payment_currency"],
            "destino_banco": transactions[index]["to_bank"],
            "destino_cuenta": transactions[index]["to_account"],
            "es_lavado_real": bool(transactions[index]["is_laundering"]),
        }
        for index in order
    ]


def build_explanation(
    sender_id: str,
    transactions: list[dict],
    stage_a: StageAResult,
    stage_b: StageBResult | None,
    *,
    top_k: int = 3,
) -> str:
    veredicto_a = "genera una ALERTA" if stage_a.is_anomalous else "se clasifica como comportamiento NORMAL"
    parrafo = (
        f"El remitente {sender_id} tiene {len(transactions)} transacciones retenidas en su "
        f"historial. La Etapa A calculó un error de reconstrucción de {stage_a.score:.4f} "
        f"contra un umbral de validación de {stage_a.threshold:.4f} (F1-óptimo), por lo que "
        f"la señal de normalidad {veredicto_a}."
    )

    if stage_b is not None:
        attention_source = stage_b.contributions
        fuente = "el clasificador de la Etapa B"
    else:
        attention_source = stage_a.attention
        fuente = "la Etapa A"
    contributions = top_contributions(attention_source, transactions, top_k=top_k)
    detalle = "; ".join(
        f"{item['timestamp']} por {item['monto_pagado']:.2f} {item['moneda_pago']} "
        f"hacia banco {item['destino_banco']} cuenta {item['destino_cuenta']} "
        f"(peso {item['peso_atención']:.2f})"
        for item in contributions
    )
    parrafo += f" Según {fuente}, las transacciones que más influyeron fueron: {detalle}."

    if stage_b is not None:
        veredicto_b = "ALERTA" if stage_b.is_anomalous else "NORMAL"
        parrafo += (
            f" La Etapa B estimó una probabilidad de lavado de {stage_b.probability:.1%} "
            f"contra un umbral de {stage_b.threshold:.1%}, con veredicto final {veredicto_b}."
        )
    else:
        parrafo += " La Etapa B todavía no está integrada; esta explicación refleja únicamente la señal de la Etapa A."
    return parrafo
