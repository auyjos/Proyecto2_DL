"""Etapa A: autoencoder de secuencias, umbral y checkpoint reutilizable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.stage_a import (
    StageAConfig,
    anomaly_score,
    load_checkpoint,
    masked_reconstruction_error,
    save_checkpoint,
    score_loader,
    select_threshold,
    train_stage_a,
)

FEATURE_DIM = 6
MAX_LEN = 5


class _ToyDataset(Dataset):
    def __init__(self, x: np.ndarray, mask: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.mask = mask
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict:
        return {"x": self.x[index], "mask": self.mask[index], "y": self.y[index], "index": index}


def _collate(samples: list[dict]) -> dict:
    return {
        "x": torch.stack([torch.from_numpy(sample["x"]) for sample in samples]),
        "mask": torch.stack([torch.from_numpy(sample["mask"]) for sample in samples]),
        "y": torch.tensor([sample["y"] for sample in samples], dtype=torch.float32),
        "sender_id": [f"S{sample['index']}" for sample in samples],
        "entity_id": [f"E{sample['index']}" for sample in samples],
    }


def _make_loader(count: int, *, rng: np.random.Generator, shuffle: bool, positive_fraction: float = 0.0) -> DataLoader:
    lengths = rng.integers(2, MAX_LEN + 1, size=count)
    x = np.zeros((count, MAX_LEN, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((count, MAX_LEN), dtype=bool)
    for index, length in enumerate(lengths):
        x[index, :length] = rng.normal(size=(length, FEATURE_DIM)).astype(np.float32)
        mask[index, :length] = True
    n_positive = int(round(count * positive_fraction))
    y = np.array([1.0] * n_positive + [0.0] * (count - n_positive), dtype=np.float32)
    if n_positive:
        # Los positivos se generan con una escala distinta para que el
        # error de reconstrucción sea separable y el umbral tenga sentido.
        x[:n_positive] *= 6.0
    dataset = _ToyDataset(x, mask, y)
    return DataLoader(dataset, batch_size=4, shuffle=shuffle, collate_fn=_collate)


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
    from src.models.stage_a import SequenceAutoencoder

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


def test_train_stage_a_runs_and_selects_best_checkpoint_by_validation_ap() -> None:
    rng = np.random.default_rng(1)
    loaders = {
        "train_normal": _make_loader(40, rng=rng, shuffle=True),
        "validation": _make_loader(20, rng=rng, shuffle=False, positive_fraction=0.2),
    }
    model, history = train_stage_a(loaders, _tiny_config(), epochs=3, device="cpu", log_every=0)

    assert len(history.train_loss) == 3
    assert len(history.validation_average_precision) == 3
    assert all(np.isfinite(value) for value in history.train_loss)

    validation_scores = score_loader(model, loaders["validation"], torch.device("cpu"))
    assert validation_scores["score"].shape == validation_scores["y"].shape
    threshold_info = select_threshold(validation_scores["score"], validation_scores["y"])
    assert 0.0 <= threshold_info.f1 <= 1.0


def test_select_threshold_separates_perfectly_scored_classes() -> None:
    scores = np.array([0.1, 0.2, 0.15, 5.0, 6.0, 0.3])
    labels = np.array([0, 0, 0, 1, 1, 0])
    info = select_threshold(scores, labels)
    assert info.f1 == 1.0
    assert info.precision == 1.0
    assert info.recall == 1.0
    assert 0.3 < info.threshold <= 5.0


def test_select_threshold_requires_at_least_one_positive() -> None:
    import pytest

    with pytest.raises(ValueError):
        select_threshold(np.array([0.1, 0.2]), np.array([0, 0]))


def test_checkpoint_roundtrip_preserves_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    loaders = {
        "train_normal": _make_loader(24, rng=rng, shuffle=True),
        "validation": _make_loader(12, rng=rng, shuffle=False, positive_fraction=0.25),
    }
    config = _tiny_config()
    model, _ = train_stage_a(loaders, config, epochs=1, device="cpu", log_every=0)
    validation_scores = score_loader(model, loaders["validation"], torch.device("cpu"))
    threshold_info = select_threshold(validation_scores["score"], validation_scores["y"])

    checkpoint_path = tmp_path / "stage_a.pt"
    save_checkpoint(checkpoint_path, model, config, threshold_info, feature_names=[f"f{i}" for i in range(FEATURE_DIM)])
    loaded_model, loaded_config, loaded_threshold, extra = load_checkpoint(checkpoint_path)

    assert loaded_config == config
    assert loaded_threshold.threshold == threshold_info.threshold
    assert extra["feature_names"] == [f"f{i}" for i in range(FEATURE_DIM)]

    batch = next(iter(loaders["validation"]))
    original_scores = anomaly_score(model, batch, torch.device("cpu"))
    reloaded_scores = anomaly_score(loaded_model, batch, torch.device("cpu"))
    np.testing.assert_allclose(original_scores, reloaded_scores, atol=1e-6)
