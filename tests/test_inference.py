"""Integración de inferencia: Etapa A y Etapa B end-to-end, explicación en lenguaje natural."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.inference import build_explanation, load_stage_b, run_stage_a, run_stage_b, top_contributions
from src.models.stage_a import SequenceAutoencoder, StageAConfig, ThresholdInfo, save_checkpoint
from src.models.stage_b import StageBClassifier, StageBConfig
from src.models.stage_b import save_checkpoint as save_stage_b_checkpoint

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


def _tiny_stage_a_config() -> StageAConfig:
    return StageAConfig(
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


def _tiny_stage_a_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    config = _tiny_stage_a_config()
    model = SequenceAutoencoder(config)
    threshold_info = ThresholdInfo(threshold=0.05, average_precision=0.5, precision=0.5, recall=0.5, f1=0.5)
    path = tmp_path / "stage_a.pt"
    save_checkpoint(path, model, config, threshold_info)
    return path


def _tiny_stage_b_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(1)
    stage_a_config = _tiny_stage_a_config()
    config = StageBConfig(hidden_dim=8, dropout=0.0, use_stage_a_score=True, seed=1)
    model = StageBClassifier(stage_a_config, config)
    threshold_info = ThresholdInfo(threshold=0.6, average_precision=0.4, precision=0.4, recall=0.4, f1=0.4)
    path = tmp_path / "stage_b.pt"
    save_stage_b_checkpoint(path, model, stage_a_config, config, threshold_info, transferred=True)
    return path


def test_run_stage_a_aligns_attention_with_valid_transactions(tmp_path: Path) -> None:
    from src.models.stage_a import load_checkpoint

    checkpoint_path = _tiny_stage_a_checkpoint(tmp_path)
    model, _, threshold_info, _ = load_checkpoint(checkpoint_path)
    batch = _make_batch(length=2)

    result = run_stage_a(model, threshold_info, batch)

    assert result.attention.shape == (2,)
    np.testing.assert_allclose(result.attention.sum(), 1.0, atol=1e-4)
    assert result.threshold == threshold_info.threshold
    assert result.is_anomalous == (result.score > threshold_info.threshold)


def test_load_stage_b_returns_none_when_checkpoint_missing(tmp_path: Path) -> None:
    assert load_stage_b(tmp_path / "does_not_exist.pt") is None


def test_run_stage_b_returns_none_without_model() -> None:
    batch = _make_batch(length=2)
    assert run_stage_b(None, None, None, batch) is None


def test_run_stage_b_produces_probability_and_contributions(tmp_path: Path) -> None:
    checkpoint_path = _tiny_stage_b_checkpoint(tmp_path)
    loaded = load_stage_b(checkpoint_path)
    assert loaded is not None
    model, stage_a_config, threshold_info, extra = loaded
    assert extra["transferred"] is True

    torch.manual_seed(2)
    stage_a_model = SequenceAutoencoder(stage_a_config)
    batch = _make_batch(length=2)
    result = run_stage_b(model, stage_a_model, threshold_info, batch)

    assert result is not None
    assert 0.0 <= result.probability <= 1.0
    assert result.contributions.shape == (2,)
    assert result.is_anomalous == (result.probability > threshold_info.threshold)


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


def test_build_explanation_uses_stage_b_contributions_when_available() -> None:
    stage_a_attention = np.array([0.9, 0.1])
    stage_b_attention = np.array([0.2, 0.8])
    transactions = _transactions(2)
    stage_a = type(
        "StageAResult",
        (),
        {"score": 0.5, "threshold": 0.1, "is_anomalous": True, "average_precision": 0.3, "attention": stage_a_attention},
    )()
    stage_b = type(
        "StageBResult",
        (),
        {"probability": 0.9, "threshold": 0.6, "is_anomalous": True, "contributions": stage_b_attention},
    )()

    explanation = build_explanation("S1", transactions, stage_a, stage_b, top_k=1)

    assert "90.0%" in explanation
    # con top_k=1, la Etapa B debe describir la posición 1 (peso 0.8), no la 0 (peso 0.9 de la Etapa A).
    assert transactions[1]["to_account"] in explanation
