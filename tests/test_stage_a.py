"""Etapa A: arquitectura del autoencoder de secuencias."""

from __future__ import annotations

import numpy as np
import torch

from src.models.stage_a import SequenceAutoencoder, StageAConfig, masked_reconstruction_error

FEATURE_DIM = 6
MAX_LEN = 5


def _tiny_config() -> StageAConfig:
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


def test_forward_respects_shapes_and_ignores_padding_content() -> None:
    torch.manual_seed(0)
    model = SequenceAutoencoder(_tiny_config()).eval()
    x = torch.randn(3, MAX_LEN, FEATURE_DIM)
    mask = torch.zeros(3, MAX_LEN, dtype=torch.bool)
    mask[:, :2] = True

    with torch.no_grad():
        output_a = model(x, mask)
        error_a = masked_reconstruction_error(output_a["x_hat"], x, mask)
        x_perturbed = x.clone()
        x_perturbed[:, 2:] += 100.0
        output_b = model(x_perturbed, mask)
        error_b = masked_reconstruction_error(output_b["x_hat"], x_perturbed, mask)

    assert output_a["x_hat"].shape == x.shape
    assert output_a["z"].shape == (3, 8)
    assert output_a["attention_weights"].shape == (3, MAX_LEN)
    # El contenido en posiciones enmascaradas no debe afectar el score.
    np.testing.assert_allclose(error_a.numpy(), error_b.numpy(), atol=1e-4)


def test_masked_reconstruction_error_is_zero_for_perfect_reconstruction() -> None:
    x = torch.randn(2, MAX_LEN, FEATURE_DIM)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    error = masked_reconstruction_error(x.clone(), x, mask)
    np.testing.assert_allclose(error.numpy(), np.zeros(2), atol=1e-6)
