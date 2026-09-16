"""Datos sintéticos compartidos por test_stage_a.py y test_stage_b.py.

No es un archivo de test (pytest lo ignora por no empezar con ``test_``);
existe para que ambas suites no dupliquen el mismo dataset/collate/loader
y por lo tanto no puedan divergir silenciosamente del contrato real de
``src.data.sequences`` (``x``, ``mask``, ``y``, ``sender_id``, ``entity_id``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FEATURE_DIM = 6
MAX_LEN = 5


class ToyDataset(Dataset):
    def __init__(self, x: np.ndarray, mask: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.mask = mask
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> dict:
        return {"x": self.x[index], "mask": self.mask[index], "y": self.y[index], "index": index}


def collate_toy_batch(samples: list[dict]) -> dict:
    return {
        "x": torch.stack([torch.from_numpy(sample["x"]) for sample in samples]),
        "mask": torch.stack([torch.from_numpy(sample["mask"]) for sample in samples]),
        "y": torch.tensor([sample["y"] for sample in samples], dtype=torch.float32),
        "sender_id": [f"S{sample['index']}" for sample in samples],
        "entity_id": [f"E{sample['index']}" for sample in samples],
    }


def make_toy_loader(
    count: int, *, rng: np.random.Generator, shuffle: bool, positive_fraction: float = 0.0
) -> DataLoader:
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
    dataset = ToyDataset(x, mask, y)
    return DataLoader(dataset, batch_size=4, shuffle=shuffle, collate_fn=collate_toy_batch)
