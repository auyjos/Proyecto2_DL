"""Integración de inferencia: Etapa A end-to-end y explicación en lenguaje natural."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.inference import build_explanation, run_stage_a, run_stage_b, top_contributions
from src.models.stage_a import SequenceAutoencoder, StageAConfig, ThresholdInfo, save_checkpoint

FEATURE_DIM = 4
MAX_LEN = 3


def _make_batch(length: int) -> dict:
    x = torch.zeros(1, MAX_LEN, FEATURE_DIM)
    mask = torch.zeros(1, MAX_LEN, dtype=torch.bool)
    x[0, :length] = torch.randn(length, FEATURE_DIM)
    mask[0, :length] = True
    return {"x": x, "mask": mask}


def _transactions(length: int) -> list[dict]:
    return [
        {
            "timestamp": f"2026/09/1{i} 10:00",
            "amount_paid": 100.0 + i,
            "payment_currency": "US Dollar",
            "to_bank": "099",
            "to_account": f"D{i}",
            "is_laundering": 0,
        }
        for i in range(length)
    ]


def _tiny_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    config = StageAConfig(
        feature_dim=FEATURE_DIM,
        max_len=MAX_LEN,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        seed=0,
    )
    model = SequenceAutoencoder(config)
    threshold_info = ThresholdInfo(threshold=0.05, average_precision=0.5, precision=0.5, recall=0.5, f1=0.5)
    path = tmp_path / "stage_a.pt"
    save_checkpoint(path, model, config, threshold_info)
    return path


def test_run_stage_a_aligns_attention_with_valid_transactions(tmp_path: Path) -> None:
    from src.models.stage_a import load_checkpoint

    checkpoint_path = _tiny_checkpoint(tmp_path)
    model, _, threshold_info, _ = load_checkpoint(checkpoint_path)
    batch = _make_batch(length=2)

    result = run_stage_a(model, threshold_info, batch)

    assert result.attention.shape == (2,)
    np.testing.assert_allclose(result.attention.sum(), 1.0, atol=1e-4)
    assert result.threshold == threshold_info.threshold
    assert result.is_anomalous == (result.score > threshold_info.threshold)


def test_run_stage_b_placeholder_returns_none() -> None:
    assert run_stage_b() is None


def test_top_contributions_orders_by_attention_descending() -> None:
    attention = np.array([0.1, 0.6, 0.3])
    transactions = _transactions(3)

    contributions = top_contributions(attention, transactions, top_k=2)

    assert [item["posición"] for item in contributions] == [1, 2]
    assert contributions[0]["peso_atención"] == 0.6


def test_build_explanation_mentions_alert_and_pending_stage_b() -> None:
    attention = np.array([0.2, 0.8])
    transactions = _transactions(2)
    stage_a = type(
        "StageAResult",
        (),
        {"score": 0.5, "threshold": 0.1, "is_anomalous": True, "average_precision": 0.3, "attention": attention},
    )()

    explanation = build_explanation("S1", transactions, stage_a, None, top_k=1)

    assert "ALERTA" in explanation
    assert "Etapa B" in explanation
    assert "0.5000" in explanation
