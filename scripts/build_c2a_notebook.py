"""Inserta la sección C2A (Jose Ruiz) en notebooks/proyecto2.ipynb.

Idempotente por diseño: si la celda reservada de C2A (el placeholder de una
sola línea que dejó José Auyón) ya no está presente, el script se detiene en
vez de duplicar la sección. Para regenerar C2A desde cero, restaurar primero
el placeholder original o editar el notebook manualmente.
"""

import nbformat as nbf

PATH = "notebooks/proyecto2.ipynb"
nb = nbf.read(PATH, as_version=4)

PLACEHOLDER_MARKER = "Sección reservada para la implementación y resultados de Jose Ruiz"
if PLACEHOLDER_MARKER not in nb.cells[20].source:
    raise SystemExit(
        "La celda 20 ya no contiene el placeholder original de C2A; "
        "este script ya se ejecutó o el notebook cambió. Nada que hacer."
    )
assert nb.cells[20].cell_type == "markdown"
assert nb.cells[20].source.startswith("# C2A")

md_intro = nbf.v4.new_markdown_cell(
    "# C2A — Etapa A: aprendizaje de la normalidad\n"
    "\n"
    "**Autor de esta sección: Jose Ruiz.**\n"
    "\n"
    "## Decisiones de diseño\n"
    "\n"
    "La Etapa A debe procesar secuencias de longitud variable, comprimirlas en una "
    "representación y reconstruirlas, sin usar `y`. Se implementó un **autoencoder "
    "Transformer con máscara de padding** (`src/models/stage_a.py`):\n"
    "\n"
    "- **Encoder:** proyección lineal de las 33 features a `d_model=64`, codificación "
    "posicional senoidal y `nn.TransformerEncoder` (2 capas, 4 cabezas) con "
    "`src_key_padding_mask` para ignorar el padding de `mask`.\n"
    "- **Pooling por atención:** un query aprendido atiende sobre las posiciones válidas "
    "del encoder para producir `z`, el vector comprimido del comportamiento del remitente.\n"
    "- **Decoder:** `z` se difunde a las 64 posiciones, se le suma la misma codificación "
    "posicional y otro `nn.TransformerEncoder` (2 capas) reconstruye `x_hat`.\n"
    "- **Score de anomalía:** error cuadrático medio por secuencia, enmascarado para "
    "ignorar el padding (`masked_reconstruction_error`).\n"
    "\n"
    "Se eligió un Transformer y no un GRU/LSTM porque (a) la Etapa B reutilizará este "
    "encoder por transfer learning, y su atención puede servir como insumo adicional de "
    "interpretabilidad; y (b) con secuencias cortas (máx. 64 pasos) el costo cuadrático de "
    "la atención es irrelevante frente al beneficio de capturar dependencias entre "
    "transacciones no necesariamente consecutivas (p. ej. fragmentación de montos con "
    "transacciones intercaladas).\n"
    "\n"
    "**Entrenamiento:** `train_stage_a` itera exclusivamente `loaders[\"train_normal\"]` "
    "(34,746 secuencias, todas `y=0`). La AUC-PR de validación se calcula cada época solo "
    "para elegir el mejor checkpoint — nunca participa en el gradiente — porque con ~0.72% "
    "de positivos en validación (54/7,500) la pérdida de reconstrucción por sí sola no "
    "distingue si el modelo está aprendiendo algo útil para separar clases.\n"
    "\n"
    "**Umbral:** se barre la curva precisión-recall de validación y se elige el punto que "
    "maximiza F1 (`select_threshold`). Con esta prevalencia, exactitud o un percentil fijo "
    "no son justificables — casi cualquier umbral tiene accuracy >99% con recall≈0 — así "
    "que se reporta F1 como métrica principal, y AUC-PR como medida de separabilidad "
    "independiente del umbral."
)

code_setup = nbf.v4.new_code_cell(
    "import torch\n"
    "\n"
    "from src.models.stage_a import (\n"
    "    StageAConfig,\n"
    "    anomaly_score,\n"
    "    resolve_device,\n"
    "    save_checkpoint,\n"
    "    score_loader,\n"
    "    select_threshold,\n"
    "    train_stage_a,\n"
    ")\n"
    "from sklearn.metrics import confusion_matrix, precision_recall_curve\n"
    "\n"
    "REPORT_DIR_C2A = ROOT / \"reports\" / \"c2a\"\n"
    "REPORT_DIR_C2A.mkdir(parents=True, exist_ok=True)\n"
    "CHECKPOINT_PATH = ROOT / \"artifacts\" / \"checkpoints\" / \"stage_a.pt\"\n"
    "\n"
    "stage_a_config = StageAConfig(feature_dim=len(manifest[\"feature_names\"]), max_len=manifest[\"configuration\"][\"max_len\"], seed=42)\n"
    "print(f\"device: {resolve_device()} · feature_dim={stage_a_config.feature_dim} · max_len={stage_a_config.max_len}\")"
)

code_train = nbf.v4.new_code_cell(
    "import time\n"
    "\n"
    "start = time.perf_counter()\n"
    "stage_a_model, stage_a_history = train_stage_a(loaders, stage_a_config, epochs=10, device=None, log_every=1)\n"
    "stage_a_train_seconds = time.perf_counter() - start\n"
    "print(f\"\\nEntrenamiento completo: {stage_a_train_seconds:.1f} s · mejor época por AUC-PR de validación: {stage_a_history.best_epoch}\")"
)

md_curves = nbf.v4.new_markdown_cell(
    "## Curvas de entrenamiento\n"
    "\n"
    "La pérdida de reconstrucción (train_normal) baja monótonamente, pero la AUC-PR de "
    "validación alcanza su máximo temprano (época 2) y luego decae aunque la reconstrucción "
    "siga mejorando en general. Esto es honesto y esperable: el autoencoder se vuelve mejor "
    "reconstruyendo secuencias *en general*, incluidas algunas con patrones de lavado que no "
    "son tan distintas de la normalidad en las features disponibles. `train_stage_a` ya "
    "selecciona automáticamente el checkpoint de mejor AUC-PR, no el de menor pérdida final."
)

code_curves = nbf.v4.new_code_cell(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout=\"constrained\")\n"
    "epochs_x = range(len(stage_a_history.train_loss))\n"
    "axes[0].plot(epochs_x, stage_a_history.train_loss, marker=\"o\", color=\"#2870a8\")\n"
    "axes[0].set(title=\"Pérdida de reconstrucción (train_normal)\", xlabel=\"Época\", ylabel=\"MSE enmascarado\")\n"
    "axes[1].plot(epochs_x, stage_a_history.validation_average_precision, marker=\"o\", color=\"#c36724\")\n"
    "axes[1].axvline(stage_a_history.best_epoch, color=\"#c43c39\", linestyle=\"--\", label=f\"mejor época ({stage_a_history.best_epoch})\")\n"
    "axes[1].set(title=\"AUC-PR de validación (solo selección de checkpoint)\", xlabel=\"Época\", ylabel=\"Average Precision\")\n"
    "axes[1].legend(frameon=False)\n"
    "for ax in axes:\n"
    "    ax.spines[[\"top\", \"right\"]].set_visible(False)\n"
    "fig.suptitle(\"Etapa A · entrenamiento sobre normalidad\")\n"
    "fig.savefig(REPORT_DIR_C2A / \"curvas_entrenamiento.png\", dpi=160)\n"
    "plt.show()"
)

md_threshold = nbf.v4.new_markdown_cell(
    "## Umbral y evaluación\n"
    "\n"
    "El umbral se elige **solo con el split de validación**; el conjunto de prueba se usa "
    "después, una única vez, para reportar el resultado final con ese umbral ya fijo."
)

code_threshold = nbf.v4.new_code_cell(
    "device = resolve_device()\n"
    "validation_scores = score_loader(stage_a_model, loaders[\"validation\"], device)\n"
    "threshold_info = select_threshold(validation_scores[\"score\"], validation_scores[\"y\"])\n"
    "\n"
    "display(pd.DataFrame([{\n"
    "    \"Umbral (F1 óptimo)\": threshold_info.threshold,\n"
    "    \"AUC-PR (validación)\": threshold_info.average_precision,\n"
    "    \"Precisión\": threshold_info.precision,\n"
    "    \"Recall\": threshold_info.recall,\n"
    "    \"F1\": threshold_info.f1,\n"
    "}]))\n"
    "\n"
    "baseline_ap = validation_scores[\"y\"].mean()\n"
    "print(f\"AUC-PR de una alerta aleatoria (prevalencia): {baseline_ap:.4f} · Etapa A: {threshold_info.average_precision:.4f} \"\n"
    "      f\"({threshold_info.average_precision / baseline_ap:.1f}x)\")"
)

code_pr_curve = nbf.v4.new_code_cell(
    "precision, recall, _ = precision_recall_curve(validation_scores[\"y\"], validation_scores[\"score\"])\n"
    "fig, ax = plt.subplots(figsize=(5.5, 4.5), layout=\"constrained\")\n"
    "ax.plot(recall, precision, color=\"#2870a8\")\n"
    "ax.scatter([threshold_info.recall], [threshold_info.precision], color=\"#c43c39\", zorder=3, label=\"umbral elegido (F1 máx.)\")\n"
    "ax.set(title=\"Curva precisión-recall · validación\", xlabel=\"Recall\", ylabel=\"Precisión\", xlim=(0, 1), ylim=(0, 1))\n"
    "ax.legend(frameon=False)\n"
    "ax.spines[[\"top\", \"right\"]].set_visible(False)\n"
    "fig.savefig(REPORT_DIR_C2A / \"curva_precision_recall.png\", dpi=160)\n"
    "plt.show()"
)

code_test_eval = nbf.v4.new_code_cell(
    "test_scores = score_loader(stage_a_model, loaders[\"test\"], device)\n"
    "test_pred = (test_scores[\"score\"] > threshold_info.threshold).astype(int)\n"
    "tn, fp, fn, tp = confusion_matrix(test_scores[\"y\"], test_pred).ravel()\n"
    "test_precision = tp / (tp + fp) if (tp + fp) else 0.0\n"
    "test_recall = tp / (tp + fn) if (tp + fn) else 0.0\n"
    "test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (test_precision + test_recall) else 0.0\n"
    "\n"
    "display(pd.DataFrame([\n"
    "    {\"\": \"Predicho normal\", \"Real normal\": tn, \"Real lavado\": fn},\n"
    "    {\"\": \"Predicho alerta\", \"Real normal\": fp, \"Real lavado\": tp},\n"
    "]).set_index(\"\"))\n"
    "print(f\"Prueba (umbral fijo de validación) · Precisión={test_precision:.3f} · Recall={test_recall:.3f} · F1={test_f1:.3f} \"\n"
    "      f\"· detecta {tp}/{tp+fn} casos de lavado revisando {tp+fp} alertas de {len(test_scores['y'])} remitentes.\")"
)

md_limits = nbf.v4.new_markdown_cell(
    "## Limitaciones honestas de la Etapa A\n"
    "\n"
    "Solo con reconstrucción no supervisada, la Etapa A detecta ~19% de los casos de lavado "
    "en prueba con ~26% de precisión en las alertas que emite (F1≈0.22). Es una señal "
    "**muy por encima del azar** (~17x la AUC-PR de una alerta aleatoria dada la prevalencia), "
    "pero claramente insuficiente por sí sola para producción: la mayoría de los casos de "
    "lavado no elevan el error de reconstrucción lo suficiente, probablemente porque, en las "
    "features actuales (montos, tiempos, categorías), varias secuencias de lavado no se ven "
    "tan distintas de la normalidad para un modelo que nunca vio una etiqueta. Esto es "
    "exactamente el argumento de diseño del proyecto: la Etapa B debe aportar valor real "
    "sobre este punto de partida, y el experimento de ablación (Gerardo) debe demostrarlo "
    "empíricamente comparando contra este resultado y contra un clasificador supervisado "
    "entrenado desde cero."
)

code_save = nbf.v4.new_code_cell(
    "save_checkpoint(\n"
    "    CHECKPOINT_PATH,\n"
    "    stage_a_model,\n"
    "    stage_a_config,\n"
    "    threshold_info,\n"
    "    feature_names=manifest[\"feature_names\"],\n"
    "    history=stage_a_history,\n"
    ")\n"
    "print(f\"Checkpoint guardado en {CHECKPOINT_PATH} · listo para transfer learning en la Etapa B.\")"
)

md_heatmap_preview = nbf.v4.new_markdown_cell(
    "## Vista previa de interpretabilidad (Etapa A)\n"
    "\n"
    "El pooling por atención de la Etapa A ya asigna un peso a cada transacción al construir "
    "`z`. Esto es una vista previa complementaria: el análisis formal de los 5 casos "
    "requeridos por el Componente 3 usa las contribuciones de la **Etapa B** (Gerardo), que "
    "tiene la señal supervisada. Aquí solo confirmamos que el mecanismo funciona extremo a "
    "extremo con `get_sender`, tal como lo necesitará el MVP."
)

code_heatmap_preview = nbf.v4.new_code_cell(
    "example_sender = str(test_scores[\"sender_id\"][test_pred == 1][0]) if (test_pred == 1).any() else str(test_scores[\"sender_id\"][0])\n"
    "example_batch = get_sender(example_sender, ARTIFACT_DIR, split=\"test\")\n"
    "stage_a_model.eval()\n"
    "with torch.no_grad():\n"
    "    example_output = stage_a_model(example_batch[\"x\"].to(device), example_batch[\"mask\"].to(device))\n"
    "example_mask = example_batch[\"mask\"][0].numpy()\n"
    "example_attention = example_output[\"attention_weights\"][0].cpu().numpy()[example_mask]\n"
    "example_frame = pd.DataFrame(example_batch[\"transactions\"])\n"
    "example_frame[\"atención_etapa_a\"] = example_attention\n"
    "example_score = float(anomaly_score(stage_a_model, example_batch, device)[0])\n"
    "\n"
    "print(f\"sender_id={example_sender} · y={int(example_batch['y'].item())} · score={example_score:.4f} · umbral={threshold_info.threshold:.4f}\")\n"
    "display(\n"
    "    example_frame[[\"timestamp\", \"to_bank\", \"to_account\", \"amount_paid\", \"payment_currency\", \"payment_format\", \"is_laundering\", \"atención_etapa_a\"]]\n"
    "    .style.background_gradient(subset=[\"atención_etapa_a\"], cmap=\"Oranges\")\n"
    ")"
)

new_cells = [
    md_intro,
    code_setup,
    code_train,
    md_curves,
    code_curves,
    md_threshold,
    code_threshold,
    code_pr_curve,
    code_test_eval,
    md_limits,
    code_save,
    md_heatmap_preview,
    code_heatmap_preview,
]

remaining_placeholder = nbf.v4.new_markdown_cell(
    "# C2B — Transfer learning, combinación y ablación\n"
    "\n"
    "Sección reservada para Gerardo Fernandez. Debe usar los mismos splits de este "
    "manifiesto, comparar contra una línea base desde cero y producir contribuciones por "
    "transacción alineadas con `transaction_ids`. El checkpoint de la Etapa A está en "
    "`artifacts/checkpoints/stage_a.pt` (cargar con `src.models.stage_a.load_checkpoint`); "
    "`stage_a_model.encoder` y `stage_a_model.pooling` son el punto de partida sugerido "
    "para transfer learning.\n"
    "\n"
    "# Interpretabilidad e integración del MVP\n"
    "\n"
    "Las filas de `get_sender` mantienen el orden del tensor, lo que permite alinear "
    "atención, heatmap y explicación. La integración final se ejecutará cuando los "
    "checkpoints y resultados de C2 estén disponibles. El MVP (`app/streamlit_app.py`, "
    "`src/inference.py`) ya integra la Etapa A completa; expone el punto de extensión "
    "`run_stage_b` para que Gerardo conecte su modelo sin tocar la interfaz."
)

nb.cells[20:21] = new_cells + [remaining_placeholder]

nbf.write(nb, PATH)
print("Notebook actualizado:", len(nb.cells), "celdas totales")
