"""MVP: detección de lavado en secuencias de remesas (Componente 4).

Consume el contrato de C1 (`src.data.sequences.get_sender`) y los
checkpoints de la Etapa A (`src.models.stage_a`) y la Etapa B
(`src.models.stage_b`) a través de `src.inference`. Si no hay checkpoint de
la Etapa B en la ruta configurada, el MVP sigue funcionando solo con la
Etapa A y lo indica explícitamente en vez de simular un resultado.

Ejecutar localmente: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sequences import get_sender  # noqa: E402
from src.inference import build_explanation, load_stage_a, load_stage_b, run_stage_a, run_stage_b, top_contributions  # noqa: E402

DEFAULT_ARTIFACT_DIR = ROOT / "app" / "data" / "c1_hi_small"
DEFAULT_CHECKPOINT_PATH = ROOT / "app" / "model" / "stage_a.pt"
DEFAULT_STAGE_B_CHECKPOINT_PATH = ROOT / "app" / "model" / "stage_b.pt"

st.set_page_config(page_title="Detección de lavado en remesas", layout="wide")


@st.cache_resource(show_spinner="Cargando checkpoint de la Etapa A...")
def _load_stage_a(checkpoint_path: str):
    return load_stage_a(checkpoint_path)


@st.cache_resource(show_spinner="Buscando checkpoint de la Etapa B...")
def _load_stage_b(checkpoint_path: str):
    return load_stage_b(checkpoint_path)


@st.cache_data(show_spinner="Cargando remitentes de prueba...")
def _load_test_sender_ids(artifact_dir: str) -> list[str]:
    splits = json.loads((Path(artifact_dir) / "splits.json").read_text(encoding="utf-8"))
    return sorted(splits["test"]["sender_ids"])


st.title("Detección de lavado de dinero en remesas — MVP")
st.caption("Componente 4 · sistema de dos etapas: autoencoder de secuencias (A) + clasificador supervisado (B).")

with st.sidebar:
    st.header("Configuración")
    artifact_dir = st.text_input("Carpeta de artefactos C1", value=str(DEFAULT_ARTIFACT_DIR))
    checkpoint_path = st.text_input("Checkpoint de la Etapa A", value=str(DEFAULT_CHECKPOINT_PATH))
    stage_b_checkpoint_path = st.text_input("Checkpoint de la Etapa B", value=str(DEFAULT_STAGE_B_CHECKPOINT_PATH))
    show_ground_truth = st.checkbox("Mostrar etiqueta real (solo demo/evaluación)", value=False)
    top_k = st.slider("Transacciones a explicar", min_value=1, max_value=10, value=3)

try:
    stage_a_model, stage_a_config, threshold_info, extra = _load_stage_a(checkpoint_path)
    test_sender_ids = _load_test_sender_ids(artifact_dir)
except (FileNotFoundError, ValueError) as exc:
    st.error(
        f"No se pudieron cargar los artefactos ({exc}). Verifica las rutas en la barra lateral; "
        "por defecto se usan las copias congeladas en app/data y app/model."
    )
    st.stop()

stage_b_loaded = _load_stage_b(stage_b_checkpoint_path)
stage_b_model, stage_b_threshold_info = (stage_b_loaded[0], stage_b_loaded[2]) if stage_b_loaded else (None, None)

sender_id = st.selectbox("Remitente (conjunto de prueba)", options=test_sender_ids)

batch = get_sender(sender_id, artifact_dir, split="test")
transactions = batch["transactions"]

stage_a_result = run_stage_a(stage_a_model, threshold_info, batch)
stage_b_result = run_stage_b(stage_b_model, stage_a_model, stage_b_threshold_info, batch)

st.subheader("Etapa A — aprendizaje de la normalidad")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transacciones", len(transactions))
col2.metric("Score de anomalía (A)", f"{stage_a_result.score:.4f}")
col3.metric("Umbral (F1-óptimo, validación)", f"{stage_a_result.threshold:.4f}")
col4.metric("Veredicto Etapa A", "ALERTA" if stage_a_result.is_anomalous else "NORMAL")

st.subheader("Etapa B — clasificación supervisada")
if stage_b_result is None:
    st.warning(
        "No se encontró un checkpoint de la Etapa B en la ruta indicada en la barra lateral. "
        "`src.inference.run_stage_b` es el punto de extensión: en cuanto exista ese checkpoint, "
        "este panel muestra la probabilidad combinada automáticamente."
    )
else:
    col5, col6, col7 = st.columns(3)
    col5.metric("Probabilidad de lavado (B)", f"{stage_b_result.probability:.1%}")
    col6.metric("Umbral (F1-óptimo, validación)", f"{stage_b_result.threshold:.1%}")
    col7.metric("Veredicto Etapa B", "ALERTA" if stage_b_result.is_anomalous else "NORMAL")

if show_ground_truth:
    st.info(f"Etiqueta real (solo referencia): {'lavado (y=1)' if batch['y'].item() == 1 else 'normal (y=0)'}")

active_attention = stage_b_result.contributions if stage_b_result is not None else stage_a_result.attention
attention_label = "Etapa B" if stage_b_result is not None else "Etapa A"

st.subheader("Secuencia de transacciones")
frame = pd.DataFrame(transactions)
frame["atención"] = active_attention
display_columns = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "atención",
]
st.caption(f"Mapa de calor calculado con los pesos de atención de la {attention_label}.")
st.dataframe(
    frame[display_columns].style.background_gradient(subset=["atención"], cmap="Oranges"),
    width="stretch",
)

st.subheader(f"Mapa de calor de contribución ({attention_label})")
st.bar_chart(frame["atención"])

st.subheader("Explicación en lenguaje natural")
explanation = build_explanation(sender_id, transactions, stage_a_result, stage_b_result, top_k=top_k)
st.write(explanation)

with st.expander("Transacciones que más influyeron (detalle)"):
    st.table(pd.DataFrame(top_contributions(active_attention, transactions, top_k=top_k)))
