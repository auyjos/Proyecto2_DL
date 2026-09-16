"""Etapa B: transfer learning desde la Etapa A, baseline y checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.stage_a import StageAConfig, train_stage_a
from src.models.stage_b import (
    StageBClassifier,
    StageBConfig,
    evaluate_stage_b,
    load_checkpoint,
    predict_batch,
    save_checkpoint,
    score_loader_stage_b,
    train_stage_b,
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


def _make_loader(count: int, *, rng: np.random.Generator, shuffle: bool, positive_fraction: float) -> DataLoader:
    lengths = rng.integers(2, MAX_LEN + 1, size=count)
    x = np.zeros((count, MAX_LEN, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((count, MAX_LEN), dtype=bool)
    for index, length in enumerate(lengths):
        x[index, :length] = rng.normal(size=(length, FEATURE_DIM)).astype(np.float32)
        mask[index, :length] = True
    n_positive = int(round(count * positive_fraction))
    y = np.array([1.0] * n_positive + [0.0] * (count - n_positive), dtype=np.float32)
    if n_positive:
        x[:n_positive] *= 6.0
    dataset = _ToyDataset(x, mask, y)
    return DataLoader(dataset, batch_size=4, shuffle=shuffle, collate_fn=_collate)


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


def _make_loaders(seed: int) -> dict[str, DataLoader]:
    rng = np.random.default_rng(seed)
    return {
        "train": _make_loader(48, rng=rng, shuffle=True, positive_fraction=0.25),
        "train_normal": _make_loader(36, rng=rng, shuffle=True, positive_fraction=0.0),
        "validation": _make_loader(20, rng=rng, shuffle=False, positive_fraction=0.2),
        "test": _make_loader(20, rng=rng, shuffle=False, positive_fraction=0.2),
    }


def test_forward_without_stage_a_score_matches_encoder_only_head() -> None:
    stage_a_config = _tiny_stage_a_config()
    config = StageBConfig(hidden_dim=8, dropout=0.0, use_stage_a_score=False, seed=0)
    model = StageBClassifier(stage_a_config, config).eval()
    x = torch.randn(2, MAX_LEN, FEATURE_DIM)
    mask = torch.zeros(2, MAX_LEN, dtype=torch.bool)
    mask[:, :3] = True

    output = model(x, mask, stage_a_score=None)

    assert output["logit"].shape == (2,)
    assert output["z"].shape == (2, stage_a_config.d_model)


def test_forward_requires_stage_a_score_when_configured() -> None:
    stage_a_config = _tiny_stage_a_config()
    config = StageBConfig(hidden_dim=8, dropout=0.0, use_stage_a_score=True, seed=0)
    model = StageBClassifier(stage_a_config, config)
    x = torch.randn(1, MAX_LEN, FEATURE_DIM)
    mask = torch.ones(1, MAX_LEN, dtype=torch.bool)

    try:
        model(x, mask, stage_a_score=None)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_load_backbone_from_stage_a_copies_encoder_weights_exactly() -> None:
    stage_a_config = _tiny_stage_a_config()
    loaders = _make_loaders(seed=1)
    stage_a_model, _ = train_stage_a(loaders, stage_a_config, epochs=1, device="cpu", log_every=0)

    config = StageBConfig(hidden_dim=8, dropout=0.0, use_stage_a_score=True, seed=0)
    classifier = StageBClassifier(stage_a_config, config)
    classifier.load_backbone_from_stage_a(stage_a_model)

    for transferred, original in zip(classifier.encoder.parameters(), stage_a_model.encoder.parameters()):
        np.testing.assert_allclose(transferred.detach().numpy(), original.detach().numpy())


def test_train_stage_b_transferred_and_baseline_run_end_to_end() -> None:
    stage_a_config = _tiny_stage_a_config()
    loaders = _make_loaders(seed=2)
    stage_a_model, _ = train_stage_a(loaders, stage_a_config, epochs=1, device="cpu", log_every=0)

    transferred_model, transferred_history = train_stage_b(
        loaders,
        stage_a_config,
        StageBConfig(hidden_dim=8, dropout=0.0, use_stage_a_score=True, seed=0),
        stage_a_model=stage_a_model,
        pos_weight=3.0,
        epochs=2,
        freeze_epochs=1,
        device="cpu",
        log_every=0,
    )
    baseline_model, baseline_history = train_stage_b(
        loaders,
        stage_a_config,
        StageBConfig(hidden_dim=8, dropout=0.0, use_stage_a_score=False, seed=0),
        stage_a_model=None,
        pos_weight=3.0,
        epochs=2,
        freeze_epochs=0,
        device="cpu",
        log_every=0,
    )

    assert len(transferred_history.train_loss) == 2
    assert transferred_history.backbone_frozen_epochs == 1
    assert baseline_history.backbone_frozen_epochs == 0

    transferred_scores = score_loader_stage_b(transferred_model, loaders["test"], torch.device("cpu"), stage_a_model)
    baseline_scores = score_loader_stage_b(baseline_model, loaders["test"], torch.device("cpu"), None)
    assert transferred_scores["probability"].shape == baseline_scores["probability"].shape
    assert ((transferred_scores["probability"] >= 0) & (transferred_scores["probability"] <= 1)).all()


def test_evaluate_stage_b_reports_consistent_metrics() -> None:
    stage_a_config = _tiny_stage_a_config()
    loaders = _make_loaders(seed=3)
    stage_a_model, _ = train_stage_a(loaders, stage_a_config, epochs=1, device="cpu", log_every=0)
    model, _ = train_stage_b(
        loaders,
        stage_a_config,
        StageBConfig(hidden_dim=8, dropout=0.0, use_stage_a_score=True, seed=0),
        stage_a_model=stage_a_model,
        pos_weight=3.0,
        epochs=1,
        freeze_epochs=0,
        device="cpu",
        log_every=0,
    )
    result = evaluate_stage_b(model, loaders, torch.device("cpu"), stage_a_model)

    assert 0.0 <= result["test_precision"] <= 1.0
    assert 0.0 <= result["test_recall"] <= 1.0
    assert 0.0 <= result["test_f1"] <= 1.0
    tn, fp, fn, tp = (
        result["test_confusion_matrix"][0][0],
        result["test_confusion_matrix"][0][1],
        result["test_confusion_matrix"][1][0],
        result["test_confusion_matrix"][1][1],
    )
    assert tn + fp + fn + tp == len(loaders["test"].dataset)


def test_checkpoint_roundtrip_preserves_predictions(tmp_path: Path) -> None:
    stage_a_config = _tiny_stage_a_config()
    loaders = _make_loaders(seed=4)
    stage_a_model, _ = train_stage_a(loaders, stage_a_config, epochs=1, device="cpu", log_every=0)
    config = StageBConfig(hidden_dim=8, dropout=0.0, use_stage_a_score=True, seed=0)
    model, history = train_stage_b(
        loaders,
        stage_a_config,
        config,
        stage_a_model=stage_a_model,
        pos_weight=3.0,
        epochs=1,
        freeze_epochs=0,
        device="cpu",
        log_every=0,
    )
    result = evaluate_stage_b(model, loaders, torch.device("cpu"), stage_a_model)

    checkpoint_path = tmp_path / "stage_b.pt"
    save_checkpoint(checkpoint_path, model, stage_a_config, config, result["threshold"], transferred=True, history=history)
    loaded_model, loaded_stage_a_config, loaded_config, loaded_threshold, extra = load_checkpoint(checkpoint_path)

    assert loaded_stage_a_config == stage_a_config
    assert loaded_config == config
    assert loaded_threshold.threshold == result["threshold"].threshold
    assert extra["transferred"] is True

    batch = next(iter(loaders["test"]))
    single_batch = {key: (value[:1] if torch.is_tensor(value) else value[:1]) for key, value in batch.items()}
    original = predict_batch(model, single_batch, torch.device("cpu"), stage_a_model)
    reloaded = predict_batch(loaded_model, single_batch, torch.device("cpu"), stage_a_model)
    np.testing.assert_allclose(original["probability"], reloaded["probability"], atol=1e-6)
