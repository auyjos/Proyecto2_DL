"""Integración de inferencia para el MVP: Etapa A lista, Etapa B como extensión.

``run_stage_a`` carga un checkpoint de ``src.models.stage_a`` y produce el
score de anomalía, el veredicto contra el umbral y los pesos de atención del
pooling (alineados con las transacciones de ``get_sender``, en el mismo
orden). ``run_stage_b`` es el punto de extensión para Gerardo: hoy no existe
checkpoint de la Etapa B, así que devuelve ``None`` y el MVP sigue
funcionando solo con la Etapa A. Cuando Gerardo publique su checkpoint y
función de predicción, basta reemplazar el cuerpo de ``run_stage_b`` sin
tocar ``app/streamlit_app.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.stage_a import SequenceAutoencoder, StageAConfig, ThresholdInfo, anomaly_score, load_checkpoint


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
    contributions: np.ndarray


def load_stage_a(checkpoint_path: str | Path) -> tuple[SequenceAutoencoder, StageAConfig, ThresholdInfo, dict]:
    return load_checkpoint(checkpoint_path)


def run_stage_a(model: SequenceAutoencoder, threshold_info: ThresholdInfo, batch: dict) -> StageAResult:
    """Score, veredicto y atención para un batch de tamaño 1 de ``get_sender``."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        output = model(batch["x"].to(device), batch["mask"].to(device))
    score = float(anomaly_score(model, batch, device)[0])
    mask = batch["mask"][0].numpy()
    attention = output["attention_weights"][0].detach().cpu().numpy()[mask]
    return StageAResult(
        score=score,
        threshold=threshold_info.threshold,
        is_anomalous=score > threshold_info.threshold,
        average_precision=threshold_info.average_precision,
        attention=attention,
    )


def run_stage_b(*_args, **_kwargs) -> StageBResult | None:
    """Placeholder para la Etapa B (Gerardo Fernandez).

    Debe devolver ``StageBResult(probability=..., contributions=<array por
    transacción, mismo orden y longitud que las filas de ``batch["transactions"]``
    de ``get_sender``>)``, o ``None`` mientras el checkpoint de la Etapa B no
    exista, para que el MVP siga mostrando únicamente la Etapa A sin romperse.
    """
    return None


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
    veredicto = "genera una ALERTA" if stage_a.is_anomalous else "se clasifica como comportamiento NORMAL"
    contributions = top_contributions(stage_a.attention, transactions, top_k=top_k)
    detalle = "; ".join(
        f"{item['timestamp']} por {item['monto_pagado']:.2f} {item['moneda_pago']} "
        f"hacia banco {item['destino_banco']} cuenta {item['destino_cuenta']} "
        f"(peso {item['peso_atención']:.2f})"
        for item in contributions
    )
    parrafo = (
        f"El remitente {sender_id} tiene {len(transactions)} transacciones retenidas en su "
        f"historial. La Etapa A calculó un error de reconstrucción de {stage_a.score:.4f} "
        f"contra un umbral de validación de {stage_a.threshold:.4f} (F1-óptimo), por lo que "
        f"el sistema {veredicto}. Las transacciones que más influyeron en la representación "
        f"aprendida por el modelo, según los pesos de atención del pooling, fueron: {detalle}."
    )
    if stage_b is not None:
        parrafo += (
            f" La Etapa B estimó una probabilidad de lavado de {stage_b.probability:.1%} "
            f"combinando la representación aprendida con el clasificador supervisado."
        )
    else:
        parrafo += (
            " La Etapa B (Gerardo Fernandez) todavía no está integrada en esta rama; esta "
            "explicación refleja únicamente la señal de la Etapa A."
        )
    return parrafo
