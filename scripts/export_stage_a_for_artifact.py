"""Exportar la Etapa A a JSON para el MVP.

Genera ``app/artifact/{weights,samples,meta}.json`` a partir del checkpoint
de la Etapa A y un subconjunto curado del conjunto de prueba (todos los
positivos reales más una muestra aleatoria de negativos), para que
``app/artifact/index.html`` corra la inferencia en JavaScript sin depender
de un servidor. El forward pass en JS está validado contra esta misma
exportación: coincide con PyTorch hasta
6 cifras decimales.

Uso: ``python scripts/export_stage_a_for_artifact.py`` desde la raíz del
proyecto, con los artefactos de C1 y el checkpoint de la Etapa A ya
generados (ver README).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from src.data.sequences import get_sender
from src.models.stage_a import anomaly_score, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "c1_hi_small"
CHECKPOINT_PATH = ROOT / "app" / "model" / "stage_a.pt"
OUT_DIR = ROOT / "app" / "artifact"
NEGATIVE_SAMPLE_SIZE = 200
SEED = 42


def round_list(values, ndigits: int = 6) -> list:
    return np.round(np.asarray(values, dtype=np.float64), ndigits).tolist()


def select_sample_ids() -> list[str]:
    arrays = np.load(ARTIFACT_DIR / "sequences.npz", allow_pickle=False)
    test_mask = arrays["split_codes"] == 2
    test_sender_ids = arrays["sender_ids"][test_mask]
    test_labels = arrays["y"][test_mask]
    positive_ids = [str(s) for s, y in zip(test_sender_ids, test_labels) if y == 1]
    negative_ids = [str(s) for s, y in zip(test_sender_ids, test_labels) if y == 0]
    print(f"positivos en test: {len(positive_ids)} · negativos en test: {len(negative_ids)}")
    sampled_negatives = random.Random(SEED).sample(negative_ids, min(NEGATIVE_SAMPLE_SIZE, len(negative_ids)))
    return positive_ids + sampled_negatives


def build_sample(sender_id: str, model) -> dict:
    batch = get_sender(sender_id, ARTIFACT_DIR, split="test")
    x = batch["x"][0].numpy()
    mask = batch["mask"][0].numpy()
    score = float(anomaly_score(model, batch)[0])
    with torch.no_grad():
        output = model(batch["x"], batch["mask"])
    attention = output["attention_weights"][0].numpy()[mask].tolist()
    transactions = [
        {
            "timestamp": row["timestamp"],
            "from_bank": row["from_bank"],
            "from_account": row["from_account"],
            "to_bank": row["to_bank"],
            "to_account": row["to_account"],
            "amount_paid": float(row["amount_paid"]),
            "payment_currency": row["payment_currency"],
            "payment_format": row["payment_format"],
            "is_laundering": int(row["is_laundering"]),
        }
        for row in batch["transactions"]
    ]
    return {
        "sender_id": sender_id,
        "entity_id": batch["entity_id"][0],
        "y": int(batch["y"].item()),
        "length": int(mask.sum()),
        "x": round_list(x),
        "mask": mask.tolist(),
        "reference_score": round(score, 6),
        "reference_attention": round_list(attention),
        "transactions": transactions,
    }


def main() -> None:
    model, config, threshold_info, _extra = load_checkpoint(CHECKPOINT_PATH)
    model.eval()

    weights = {
        key: {"shape": list(tensor.shape), "data": round_list(tensor.numpy().ravel())}
        for key, tensor in model.state_dict().items()
    }

    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    sample_ids = select_sample_ids()
    print(f"total elegidos: {len(sample_ids)}")
    samples = [build_sample(sender_id, model) for sender_id in sample_ids]

    meta = {
        "config": {
            "feature_dim": config.feature_dim,
            "max_len": config.max_len,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_encoder_layers": config.num_encoder_layers,
            "num_decoder_layers": config.num_decoder_layers,
            "dim_feedforward": config.dim_feedforward,
        },
        "threshold": threshold_info.as_dict(),
        "feature_names": manifest["feature_names"],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "weights.json").write_text(json.dumps(weights, separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "samples.json").write_text(json.dumps(samples, separators=(",", ":")), encoding="utf-8")
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"weights.json: {(OUT_DIR / 'weights.json').stat().st_size:,} bytes")
    print(f"samples.json: {(OUT_DIR / 'samples.json').stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
