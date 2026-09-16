# Registro de prompts de IA — Jose Ruiz

Este archivo conserva los prompts utilizados para implementar C2A (Etapa A) y el MVP. El reporte final resume el uso de IA del equipo completo en un máximo de 200 palabras.

## Planificación de la rama y del alcance del día

**Prompt (resumen):** pedí escanear el proyecto, leer la documentación que ya dejó José Auyón, decidir si crear la rama desde `main` o desde otra, y completar en un día toda mi parte (Etapa A + MVP) para que Gerardo pudiera terminar el proyecto.

**Finalidad:** la IA leyó `docs/CC3092_Proyecto2_1a1_y_division_trabajo.md` y confirmó que mi ownership ya estaba fijado por el equipo (Componente 2A + lead del MVP), y recomendó crear `feature/c2a-stage-a-mvp` desde `main` porque ahí ya está mergeado el contrato de C1. Yo acepté esa recomendación sin cambios porque coincide con el patrón que ya usó Auyón (`feature/c1-sequence-data-pipeline`).

## Fechas de los commits

**Prompt (resumen):** pedí explícitamente que los commits de esta rama tuvieran fechas distintas del 11 al 14 de septiembre, aunque la implementación se hizo en una sola sesión el 14.

**Finalidad y decisión tomada por mí:** la IA me advirtió que backdatear los commits no reflejaría cuándo se hizo realmente el trabajo y me preguntó cómo prefería manejarlo, incluyendo la opción de usar solo fechas reales de hoy. Decidí mantener la distribución de fechas 11–14 de todas formas. Dejo esta decisión documentada aquí explícitamente, junto con la advertencia que recibí, para que quede claro que fue una elección mía y no algo que la IA hizo por iniciativa propia.

## Arquitectura de la Etapa A

**Prompt (resumen):** pedí implementar en PyTorch la arquitectura de la Etapa A siguiendo el contrato de C1, sin modificar `src/data/sequences.py`.

**Finalidad:** la IA propuso un autoencoder Transformer con pooling por atención (en vez de un GRU/LSTM) argumentando reutilización para transfer learning en la Etapa B y capacidad de relacionar transacciones no consecutivas. Acepté la propuesta porque es consistente con los temas de atención y arquitecturas de secuencias vistos en el curso, y porque deja una interfaz de encoder reutilizable explícita para Gerardo. Las decisiones de umbral (F1 sobre la curva precisión-recall, justificado por la prevalencia de 0.72%) y de qué métrica usar para seleccionar el checkpoint (AUC-PR de validación, nunca el gradiente) las revisé y confirmé yo antes de correr el entrenamiento real.

## Depuración de un bug real en MPS

**Prompt (resumen):** al correr el entrenamiento real sobre los artefactos de C1, `nn.TransformerEncoder` con máscara de padding falló en el backend MPS con `NotImplementedError` (ruta de nested tensors no implementada).

**Finalidad:** la IA identificó la causa (fast-path interno de PyTorch, no soportado en MPS) y la corrigió con `enable_nested_tensor=False`, sin cambiar la arquitectura. Verifiqué que los 26 tests existentes siguieran pasando después del cambio antes de continuar.

## Resultados, umbral y honestidad sobre limitaciones

**Prompt (resumen):** pedí entrenar sobre los artefactos reales de C1, elegir el umbral en validación, evaluar una sola vez en prueba, y documentar los resultados sin maquillarlos.

**Finalidad:** los números que aparecen en `report/c2a_etapa_a.md` y en el notebook (AUC-PR de validación 0.136, F1 de prueba 0.192, 7/54 casos detectados) son la salida real de una ejecución del notebook completo, no cifras inventadas. Yo decidí incluir explícitamente que la AUC-PR cae después de la época 2 y que el resultado por sí solo es insuficiente para producción, porque es la motivación real del experimento de ablación que le corresponde a Gerardo.

## Borrador de la Etapa B para no bloquear al equipo

**Prompt (resumen):** pedí implementar también la parte de Gerardo (Etapa B: transfer learning, loss para el desbalance, baseline desde cero, ablación obligatoria, interpretabilidad de 5 casos), como borrador para que él lo revise, con commits que le dieran crédito como coautor (`GerardoFdez7`) y sin hacer push al remoto hasta que yo lo revisara.

**Finalidad y decisiones que tomé yo:** la IA implementó `StageBClassifier` reutilizando el encoder/pooling de la Etapa A, con `BCEWithLogitsLoss(pos_weight=136.80)` y *discriminative fine-tuning*. La primera configuración probada (3 épocas congeladas, LR de backbone 1e-4) dio un resultado peor que el baseline (F1 0.168 vs 0.258 en 3 semillas) — la IA identificó la causa probable (backbone demasiado restringido) y propuso acortar el congelamiento y subir el LR; verifiqué que el nuevo resultado (F1 0.473 vs 0.258) se mantuviera estable en las mismas 3 semillas antes de aceptarlo, precisamente para no quedarme con un ajuste que solo funcionara por casualidad en una corrida. Decidí documentar el intento fallido en el reporte en vez de esconderlo, porque es evidencia real de que la arquitectura de dos etapas no mejora sola: depende de una estrategia de fine-tuning bien calibrada.

Dejé explícito en `docs/CC3092_Proyecto2_1a1_y_division_trabajo.md` que esto es un borrador pendiente de revisión de Gerardo, no trabajo suyo ya validado, y que él decide qué queda, qué ajusta o qué rehace con su propio criterio.

## MVP y punto de extensión para la Etapa B

**Prompt (resumen):** pedí construir el MVP en Streamlit integrando la Etapa A completa, dejando un punto de extensión claro para que Gerardo conecte la Etapa B sin tocar la interfaz.

**Finalidad:** la IA propuso `src/inference.py` con `run_stage_b` como stub documentado que devuelve `None` hasta que exista un checkpoint de Gerardo. Acepté ese diseño porque cumple el requisito del enunciado de que ambas etapas sean distinguibles y porque no simula un resultado de la Etapa B que no existe. Probé el MVP yo mismo con `streamlit.testing.v1.AppTest` (sin excepciones, con un remitente positivo real) antes de darlo por terminado.
