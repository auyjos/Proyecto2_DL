"""Añadir la Etapa B (transfer learning) al mismo Claude Artifact de la Etapa A.

Reutiliza los ``app/artifact/{weights,samples,meta}.json`` que ya publicó
``export_stage_a_for_artifact.py`` en vez de regenerarlos desde cero: añade
los pesos de ``StageBClassifier`` a ``weights.json`` con el prefijo
``stageB.`` (para no chocar con las claves de la Etapa A, que comparten
nombres de submódulo por ser el mismo backbone transferido), el umbral de
la Etapa B a ``meta.json``, y una probabilidad + atención de referencia por
caso a ``samples.json`` — calculadas con exactamente el mismo contrato que
``src.inference.run_stage_b`` (``predict_batch`` de
``src.models.stage_b``), para los mismos 254 remitentes ya elegidos por la
Etapa A (se leen de ``samples.json``, no se vuelve a muestrear). El batch
se reconstruye directamente desde los campos ``x``/``mask`` ya exportados
en vez de volver a llamar ``get_sender`` sobre el contrato de C1, para no
depender de que ``artifacts/c1_hi_small`` exista en la máquina donde se
corre este script.

Uso: ``python scripts/export_stage_b_for_artifact.py`` desde la raíz del
proyecto, después de haber corrido ``export_stage_a_for_artifact.py`` al
menos una vez y con el checkpoint de la Etapa B ya generado
(``app/model/stage_b.pt``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.models.stage_a import load_checkpoint as load_stage_a_checkpoint
from src.models.stage_b import load_checkpoint as load_stage_b_checkpoint
from src.models.stage_b import predict_batch

ROOT = Path(__file__).resolve().parents[1]
STAGE_A_CHECKPOINT = ROOT / "app" / "model" / "stage_a.pt"
STAGE_B_CHECKPOINT = ROOT / "app" / "model" / "stage_b.pt"
OUT_DIR = ROOT / "app" / "artifact"
WEIGHT_PREFIX = "stageB."


def round_list(values, ndigits: int = 6) -> list:
    return np.round(np.asarray(values, dtype=np.float64), ndigits).tolist()


def main() -> None:
    stage_a_model, _stage_a_config, _stage_a_threshold, _extra = load_stage_a_checkpoint(STAGE_A_CHECKPOINT)
    stage_a_model.eval()
    stage_b_model, _stage_a_config_b, stage_b_config, threshold_info, extra = load_stage_b_checkpoint(STAGE_B_CHECKPOINT)
    stage_b_model.eval()

    weights_path = OUT_DIR / "weights.json"
    meta_path = OUT_DIR / "meta.json"
    samples_path = OUT_DIR / "samples.json"

    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    for key, tensor in stage_b_model.state_dict().items():
        weights[WEIGHT_PREFIX + key] = {"shape": list(tensor.shape), "data": round_list(tensor.numpy().ravel())}
    weights_path.write_text(json.dumps(weights, separators=(",", ":")), encoding="utf-8")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["stage_b"] = {
        "config": stage_b_config.as_dict(),
        "threshold": threshold_info.as_dict(),
        "transferred": extra.get("transferred"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    for sample in samples:
        batch = {
            "x": torch.tensor([sample["x"]], dtype=torch.float32),
            "mask": torch.tensor([sample["mask"]], dtype=torch.bool),
        }
        prediction = predict_batch(stage_b_model, batch, torch.device("cpu"), stage_a_model)
        sample["reference_probability_b"] = round(float(prediction["probability"][0]), 6)
        sample["reference_attention_b"] = round_list(prediction["attention"])
    samples_path.write_text(json.dumps(samples, separators=(",", ":")), encoding="utf-8")

    print(f"weights.json ahora: {weights_path.stat().st_size:,} bytes")
    print(f"samples.json ahora: {samples_path.stat().st_size:,} bytes")
    print(f"umbral Etapa B: {threshold_info.threshold:.4f} · AP validación: {threshold_info.average_precision:.4f}")


if __name__ == "__main__":
    main()
