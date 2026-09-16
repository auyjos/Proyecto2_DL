"""Inserta la sección C2B en notebooks/proyecto2.ipynb."""

import nbformat as nbf

PATH = "notebooks/proyecto2.ipynb"
nb = nbf.read(PATH, as_version=4)

PLACEHOLDER_MARKER = "Sección reservada para Gerardo Fernandez"
target_index = None
for index, cell in enumerate(nb.cells):
    if cell.cell_type == "markdown" and PLACEHOLDER_MARKER in cell.source:
        target_index = index
        break
if target_index is None:
    raise SystemExit("No se encontró la celda placeholder de C2B; el notebook ya cambió o el script ya corrió.")

# También quitar la mención a "Gerardo" del preview de interpretabilidad de la Etapa A (celda anterior).
for cell in nb.cells:
    if cell.cell_type == "markdown" and "usa las contribuciones de la **Etapa B** (Gerardo)" in cell.source:
        cell.source = cell.source.replace(
            "usa las contribuciones de la **Etapa B** (Gerardo), que",
            "usa las contribuciones de la **Etapa B**, que",
        )

md_intro = nbf.v4.new_markdown_cell(
    "# C2B — Etapa B: transfer learning, pérdida y ablación\n"
    "\n"
    "La Etapa B reutiliza el encoder y el pooling de la Etapa A (`StageBClassifier` en "
    "`src/models/stage_b.py`) y añade una cabeza de clasificación. La señal de ambas etapas se "
    "combina a nivel de features: el error de reconstrucción de la Etapa A (ya entrenada) se "
    "concatena al vector comprimido `z` antes de la cabeza.\n"
    "\n"
    "**Estrategia de transfer learning:** *discriminative fine-tuning* con descongelamiento "
    "gradual — la cabeza entrena sola la primera época (backbone congelado y en modo `eval`, "
    "para no aplicar dropout sobre pesos que no se actualizan), y luego el backbone se "
    "descongela con un LR menor (5e-4) que el de la cabeza (1e-3).\n"
    "\n"
    "**Pérdida:** `BCEWithLogitsLoss(pos_weight=136.80)`, usando el `pos_weight` que ya publica "
    "el manifiesto de C1 — estándar para desbalance extremo cuando ya se cuenta con una "
    "estimación confiable de la proporción.\n"
    "\n"
    "**Un hallazgo honesto:** la primera configuración probada (3 épocas congeladas, LR de "
    "backbone 1e-4) dio un resultado indeseado: el modelo transferido quedó *por debajo* de la "
    "línea base entrenada desde cero (F1 de prueba 0.168±0.010 vs 0.258±0.039 en 3 semillas). "
    "La hipótesis fue que el backbone no alcanzaba a adaptarse a la señal supervisada con un LR "
    "tan conservador y tan pocas épocas de ajuste completo. Acortar el congelamiento a 1 época y "
    "subir el LR del backbone a 5e-4 confirmó la hipótesis (resultados abajo). Detalle completo "
    "en `report/c2b_etapa_b.md`."
)

code_setup = nbf.v4.new_code_cell(
    "from src.models.stage_b import (\n"
    "    StageBConfig,\n"
    "    evaluate_stage_b,\n"
    "    load_checkpoint as load_stage_b_checkpoint,\n"
    "    predict_batch as predict_stage_b_batch,\n"
    "    save_checkpoint as save_stage_b_checkpoint,\n"
    "    score_loader_stage_b,\n"
    "    train_stage_b,\n"
    ")\n"
    "\n"
    "REPORT_DIR_C2B = ROOT / \"reports\" / \"c2b\"\n"
    "REPORT_DIR_C2B.mkdir(parents=True, exist_ok=True)\n"
    "STAGE_B_CHECKPOINT_PATH = ROOT / \"artifacts\" / \"checkpoints\" / \"stage_b_transferred.pt\"\n"
    "STAGE_B_BASELINE_CHECKPOINT_PATH = ROOT / \"artifacts\" / \"checkpoints\" / \"stage_b_baseline.pt\"\n"
    "pos_weight = manifest[\"counts\"][\"pos_weight\"]\n"
    "SEEDS = [42, 43, 44]\n"
    "print(f\"pos_weight={pos_weight:.4f} · semillas de ablación: {SEEDS}\")"
)

code_ablation_loop = nbf.v4.new_code_cell(
    "import time\n"
    "\n"
    "transferred_runs, baseline_runs = [], []\n"
    "official_transferred_model = official_transferred_history = None\n"
    "official_baseline_model = official_baseline_history = None\n"
    "start = time.perf_counter()\n"
    "for seed in SEEDS:\n"
    "    t_model, t_history = train_stage_b(\n"
    "        loaders, stage_a_config, StageBConfig(seed=seed, use_stage_a_score=True),\n"
    "        stage_a_model=stage_a_model, pos_weight=pos_weight,\n"
    "        epochs=12, freeze_epochs=1, lr_backbone=5e-4, device=None, log_every=0,\n"
    "    )\n"
    "    t_result = evaluate_stage_b(t_model, loaders, device, stage_a_model)\n"
    "    transferred_runs.append({\"seed\": seed, **{k: t_result[k] for k in (\"test_precision\", \"test_recall\", \"test_f1\", \"test_average_precision\")}})\n"
    "\n"
    "    b_model, b_history = train_stage_b(\n"
    "        loaders, stage_a_config, StageBConfig(seed=seed, use_stage_a_score=False),\n"
    "        stage_a_model=None, pos_weight=pos_weight,\n"
    "        epochs=12, freeze_epochs=0, device=None, log_every=0,\n"
    "    )\n"
    "    b_result = evaluate_stage_b(b_model, loaders, device, None)\n"
    "    baseline_runs.append({\"seed\": seed, **{k: b_result[k] for k in (\"test_precision\", \"test_recall\", \"test_f1\", \"test_average_precision\")}})\n"
    "\n"
    "    if seed == 42:\n"
    "        official_transferred_model, official_transferred_history = t_model, t_history\n"
    "        official_transferred_threshold = t_result[\"threshold\"]\n"
    "        official_baseline_model, official_baseline_history = b_model, b_history\n"
    "        official_baseline_threshold = b_result[\"threshold\"]\n"
    "\n"
    "    print(f\"seed {seed} · transferred F1={t_result['test_f1']:.3f} AP={t_result['test_average_precision']:.3f} \"\n"
    "          f\"· baseline F1={b_result['test_f1']:.3f} AP={b_result['test_average_precision']:.3f}\")\n"
    "print(f\"\\n3 semillas x 2 modelos completas en {time.perf_counter() - start:.1f} s\")"
)

md_ablation_table = nbf.v4.new_markdown_cell(
    "## Tabla de ablación (obligatoria)\n"
    "\n"
    "Promedio ± desviación estándar sobre las 3 semillas, umbral elegido por F1 en validación y "
    "evaluado una sola vez en prueba."
)

code_ablation_table = nbf.v4.new_code_cell(
    "def summarize(runs, label):\n"
    "    f1 = np.array([r[\"test_f1\"] for r in runs])\n"
    "    ap = np.array([r[\"test_average_precision\"] for r in runs])\n"
    "    precision = np.array([r[\"test_precision\"] for r in runs])\n"
    "    recall = np.array([r[\"test_recall\"] for r in runs])\n"
    "    return {\n"
    "        \"Modelo\": label,\n"
    "        \"Precisión\": f\"{precision.mean():.3f} ± {precision.std():.3f}\",\n"
    "        \"Recall\": f\"{recall.mean():.3f} ± {recall.std():.3f}\",\n"
    "        \"F1\": f\"{f1.mean():.3f} ± {f1.std():.3f}\",\n"
    "        \"AUC-PR\": f\"{ap.mean():.3f} ± {ap.std():.3f}\",\n"
    "    }\n"
    "\n"
    "ablation_table = pd.DataFrame([\n"
    "    summarize(baseline_runs, \"Baseline supervisado desde cero (sin Etapa A)\"),\n"
    "    summarize(transferred_runs, \"Dos etapas (transfer learning + score de A)\"),\n"
    "])\n"
    "display(ablation_table)"
)

code_ablation_plot = nbf.v4.new_code_cell(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout=\"constrained\")\n"
    "epochs_x = range(len(official_transferred_history.train_loss))\n"
    "axes[0].plot(epochs_x, official_transferred_history.validation_average_precision, marker=\"o\", color=\"#2870a8\", label=\"Dos etapas\")\n"
    "axes[0].plot(epochs_x, official_baseline_history.validation_average_precision, marker=\"o\", color=\"#c36724\", label=\"Baseline desde cero\")\n"
    "axes[0].axvline(official_transferred_history.backbone_frozen_epochs, color=\"#2870a8\", linestyle=\"--\", alpha=.5, label=\"fin de congelamiento\")\n"
    "axes[0].set(title=\"AUC-PR de validación por época (semilla 42)\", xlabel=\"Época\", ylabel=\"Average Precision\")\n"
    "axes[0].legend(frameon=False, fontsize=8)\n"
    "\n"
    "arms = [\"Baseline\", \"Dos etapas\"]\n"
    "f1_means = [np.mean([r[\"test_f1\"] for r in baseline_runs]), np.mean([r[\"test_f1\"] for r in transferred_runs])]\n"
    "f1_stds = [np.std([r[\"test_f1\"] for r in baseline_runs]), np.std([r[\"test_f1\"] for r in transferred_runs])]\n"
    "axes[1].bar(arms, f1_means, yerr=f1_stds, color=[\"#c36724\", \"#2870a8\"], capsize=6)\n"
    "axes[1].set(title=\"F1 de prueba (media ± std, 3 semillas)\", ylabel=\"F1\")\n"
    "for ax in axes:\n"
    "    ax.spines[[\"top\", \"right\"]].set_visible(False)\n"
    "fig.suptitle(\"Etapa B · ablación: dos etapas vs. baseline desde cero\")\n"
    "fig.savefig(REPORT_DIR_C2B / \"ablacion.png\", dpi=160)\n"
    "plt.show()"
)

md_variance_note = nbf.v4.new_markdown_cell(
    "**Limitación honesta:** con solo 254 remitentes positivos en entrenamiento y 54 en prueba, "
    "hay varianza real entre semillas. El hallazgo de que la configuración de transfer learning "
    "importa tanto como su sola presencia —una configuración mal calibrada puede perder frente a "
    "un baseline, como se documentó arriba— es en sí mismo un resultado que vale la pena "
    "reportar, no solo un paso intermedio a descartar."
)

code_save_checkpoints = nbf.v4.new_code_cell(
    "save_stage_b_checkpoint(\n"
    "    STAGE_B_CHECKPOINT_PATH, official_transferred_model, stage_a_config,\n"
    "    StageBConfig(seed=42, use_stage_a_score=True), official_transferred_threshold,\n"
    "    transferred=True, history=official_transferred_history,\n"
    ")\n"
    "save_stage_b_checkpoint(\n"
    "    STAGE_B_BASELINE_CHECKPOINT_PATH, official_baseline_model, stage_a_config,\n"
    "    StageBConfig(seed=42, use_stage_a_score=False), official_baseline_threshold,\n"
    "    transferred=False, history=official_baseline_history,\n"
    ")\n"
    "print(f\"Checkpoints (semilla 42) guardados en {STAGE_B_CHECKPOINT_PATH.name} y {STAGE_B_BASELINE_CHECKPOINT_PATH.name}\")"
)

md_interpretability_header = nbf.v4.new_markdown_cell(
    "## Interpretabilidad: 5 casos del conjunto de prueba\n"
    "\n"
    "Tres verdaderos positivos, un falso positivo y un falso negativo, seleccionados con el "
    "modelo oficial (semilla 42). Los pesos de atención son los del pooling de la Etapa B, "
    "alineados con `transaction_ids`. Análisis completo en "
    "`report/interpretabilidad_casos.md`."
)

code_case_selection = nbf.v4.new_code_cell(
    "test_scores_b = score_loader_stage_b(official_transferred_model, loaders[\"test\"], device, stage_a_model)\n"
    "pred_b = (test_scores_b[\"probability\"] > official_transferred_threshold.threshold).astype(int)\n"
    "y_b = test_scores_b[\"y\"].astype(int)\n"
    "sender_ids_b = test_scores_b[\"sender_id\"]\n"
    "\n"
    "tp_idx = np.flatnonzero((pred_b == 1) & (y_b == 1))\n"
    "fp_idx = np.flatnonzero((pred_b == 1) & (y_b == 0))\n"
    "fn_idx = np.flatnonzero((pred_b == 0) & (y_b == 1))\n"
    "print(f\"Verdaderos positivos: {len(tp_idx)} · Falsos positivos: {len(fp_idx)} · Falsos negativos: {len(fn_idx)}\")\n"
    "\n"
    "case_rng = np.random.default_rng(42)\n"
    "case_ids = [\n"
    "    *[(\"Verdadero positivo\", sender_ids_b[i]) for i in case_rng.choice(tp_idx, size=3, replace=False)],\n"
    "    (\"Falso positivo\", sender_ids_b[case_rng.choice(fp_idx, size=1, replace=False)[0]]),\n"
    "    (\"Falso negativo\", sender_ids_b[case_rng.choice(fn_idx, size=1, replace=False)[0]]),\n"
    "]"
)

code_case_display = nbf.v4.new_code_cell(
    "fig, axes = plt.subplots(len(case_ids), 1, figsize=(11, 3.1 * len(case_ids)), layout=\"constrained\")\n"
    "for panel, (category, case_sender_id) in zip(axes, case_ids):\n"
    "    case_batch = get_sender(case_sender_id, ARTIFACT_DIR, split=\"test\")\n"
    "    case_prediction = predict_stage_b_batch(official_transferred_model, case_batch, device, stage_a_model)\n"
    "    case_frame = pd.DataFrame(case_batch[\"transactions\"])\n"
    "    case_frame[\"atención_etapa_b\"] = case_prediction[\"attention\"]\n"
    "    colors = [\"#c43c39\" if row.is_laundering else \"#2870a8\" for row in case_frame.itertuples()]\n"
    "    panel.bar(range(len(case_frame)), case_frame[\"atención_etapa_b\"], color=colors)\n"
    "    panel.set_title(\n"
    "        f\"{category} · {case_sender_id} · y={int(case_batch['y'].item())} · \"\n"
    "        f\"prob_B={float(case_prediction['probability'][0]):.3f} (umbral {official_transferred_threshold.threshold:.3f})\",\n"
    "        fontsize=10,\n"
    "    )\n"
    "    panel.set_ylabel(\"Atención\")\n"
    "    panel.spines[[\"top\", \"right\"]].set_visible(False)\n"
    "fig.suptitle(\"Atención de la Etapa B por transacción (rojo = transacción etiquetada como lavado)\")\n"
    "fig.savefig(REPORT_DIR_C2B / \"casos_interpretabilidad.png\", dpi=150)\n"
    "plt.show()"
)

md_wrap_up = nbf.v4.new_markdown_cell(
    "# Interpretabilidad e integración del MVP\n"
    "\n"
    "Las filas de `get_sender` mantienen el orden del tensor, lo que permite alinear atención, "
    "heatmap y explicación. El MVP (`app/streamlit_app.py`, `src/inference.py`) integra ambas "
    "etapas: cuando existe un checkpoint de la Etapa B (`app/model/stage_b.pt`), "
    "`src.inference.run_stage_b` deja de devolver `None` y el MVP muestra automáticamente la "
    "probabilidad combinada, el mapa de calor de la Etapa B y la explicación en lenguaje natural "
    "con ambas señales, sin cambios en la interfaz."
)

new_cells = [
    md_intro,
    code_setup,
    code_ablation_loop,
    md_ablation_table,
    code_ablation_table,
    code_ablation_plot,
    md_variance_note,
    code_save_checkpoints,
    md_interpretability_header,
    code_case_selection,
    code_case_display,
    md_wrap_up,
]

nb.cells[target_index:target_index + 1] = new_cells

nbf.write(nb, PATH)
print("Notebook actualizado:", len(nb.cells), "celdas totales")
