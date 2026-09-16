# CC3092 Proyecto 2 — detección de lavado en secuencias

Este proyecto académico desarrolla un prototipo de detección de lavado de dinero a partir del historial de transacciones de cada remitente. El objetivo es identificar patrones sospechosos que una operación aislada no revela, como cambios de frecuencia, montos atípicos, nuevos destinatarios y actividad en horarios inusuales. Además de producir una alerta, el sistema debe señalar qué transacciones influyeron en ella para facilitar su revisión.

## Enfoque del proyecto

El sistema se plantea en dos etapas complementarias:

1. **Aprendizaje de la normalidad:** un modelo secuencial se entrena con historiales normales y utiliza el error de reconstrucción como score de anomalía.
2. **Clasificación supervisada:** un segundo modelo aprovecha la representación aprendida para estimar la probabilidad de lavado con los ejemplos etiquetados.

Los resultados de ambas etapas se combinarán en una predicción final y se compararán contra una línea base supervisada entrenada desde cero. El proyecto también contempla interpretabilidad sobre las transacciones, un reporte ejecutivo y un MVP para consultar remitentes del conjunto de prueba.

## Datos y alcance actual

Se analizaron PaySim y las seis variantes Small, Medium y Large de IBM AML. PaySim contiene millones de transacciones, pero casi todos sus remitentes aparecen una sola vez, lo que limita el aprendizaje temporal. Por esta razón, el pipeline de modelado utiliza **IBM AML HI-Small**, que ofrece historiales más útiles y permite reproducir los experimentos dentro del presupuesto de Google Colab. Todos los datos son sintéticos y sus etiquetas representan fenómenos distintos: fraude en PaySim y lavado de dinero en IBM AML.

La implementación actual cubre el **Componente 1 (C1)**: validación de datos, selección reproducible de remitentes, construcción de secuencias, ingeniería de features, particiones sin compartir entidades, artefactos persistidos y visualizaciones. El contrato resultante sirve como entrada común para los modelos de las etapas A y B. Los CSV originales y los artefactos generados se excluyen de Git por su tamaño.

También está integrado el **Componente 2A (Etapa A)**: un autoencoder Transformer entrenado solo sobre comportamiento normal, con umbral justificado sobre la curva precisión-recall de validación. El **Componente 2B (Etapa B)** reutiliza ese encoder por transfer learning para un clasificador supervisado, con un experimento de ablación obligatorio contra un baseline entrenado desde cero (`report/c2b_etapa_b.md`) — este borrador está pendiente de revisión por el resto del equipo. El **MVP en Streamlit** integra ambas etapas: score de la Etapa A, probabilidad de la Etapa B, mapa de calor de atención y una explicación en lenguaje natural generada automáticamente.

Los principales recursos del repositorio son:

- `src/data/sequences.py`: pipeline reutilizable y contrato de DataLoaders.
- `src/models/stage_a.py`: arquitectura, entrenamiento, umbral y checkpoint de la Etapa A.
- `src/models/stage_b.py`: clasificador de la Etapa B, transfer learning y ablación.
- `src/inference.py`: integración de inferencia que usa el MVP (ambas etapas).
- `app/streamlit_app.py`: MVP funcional del Componente 4.
- `notebooks/proyecto2.ipynb`: notebook principal ejecutado y espacio de integración del equipo.
- `notebooks/comparacion_datasets.ipynb`: comparación reproducible de los datasets.
- `docs/contrato_datos_c1.md`: formas, campos y reglas que deben respetar los modelos.
- `report/c1_ingenieria_datos.md`: sección de ingeniería de datos propuesta para el reporte final.
- `report/c2a_etapa_a.md`: sección de la Etapa A y el MVP propuesta para el reporte final.
- `report/c2b_etapa_b.md`: sección de la Etapa B y la ablación propuesta para el reporte final.
- `report/interpretabilidad_casos.md`: análisis de los 5 casos requeridos por C3.

## Preparación

En PowerShell, desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Los archivos esperados para C1 son:

```text
datasets/IBM/HI-Small_Trans.csv
datasets/IBM/HI-Small_accounts.csv
```

## Construir C1

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -c "from src.data.sequences import build_artifacts; build_artifacts('datasets/IBM/HI-Small_Trans.csv', 'artifacts/c1_hi_small')"
```

La configuración predeterminada selecciona uniformemente 50,000 cuentas emisoras con semilla 42, conserva las 64 operaciones más recientes y genera splits 70/15/15 agrupados por `Entity ID`.

## Consumir el contrato

```python
from src.data.sequences import get_sender, make_dataloaders

loaders = make_dataloaders("artifacts/c1_hi_small", batch_size=128, seed=42)
batch = next(iter(loaders["train_normal"]))  # Etapa A: solo y == 0

sender = get_sender(
    "[\"021174\",\"800737690\"]",
    "artifacts/c1_hi_small",
    split="test",
)
```

Cada batch contiene `x [B,64,F]`, `mask [B,64]`, `lengths [B]`, `y [B]`, `sender_id`, `entity_id` y `transaction_ids [B,64]`. La máscara usa `True` únicamente en operaciones reales. Reconstrucción, atención y pooling deben ignorar posiciones `False`.

Los loaders disponibles son:

- `train`: conjunto completo para la etapa B y la línea base; se baraja.
- `train_normal`: subconjunto normal para la etapa A; se baraja.
- `validation` y `test`: no se barajan.

`get_sender` añade las filas originales conservadas en `transactions`, en el mismo orden que las posiciones válidas del tensor. El MVP debe cargar estos artefactos y no reajustar vocabularios ni normalización.

## Notebook y pruebas

El notebook principal de C1 es [notebooks/proyecto2.ipynb](notebooks/proyecto2.ipynb). El análisis de datasets está en [notebooks/comparacion_datasets.ipynb](notebooks/comparacion_datasets.ipynb).

En Colab, subir o clonar el proyecto en Drive, copiar los dos CSV a las rutas indicadas y ejecutar:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/ruta/al/Proyecto2_DL
!pip install -r requirements.txt
```

Después, abrir `notebooks/proyecto2.ipynb` y ejecutar todas las celdas. La ruta dentro de Drive depende de dónde guarde el equipo la carpeta.

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest
```

## Entrenar la Etapa A

```python
import json

from src.data.sequences import make_dataloaders
from src.models.stage_a import StageAConfig, train_stage_a, score_loader, select_threshold, save_checkpoint, resolve_device

manifest = json.load(open("artifacts/c1_hi_small/manifest.json"))
loaders = make_dataloaders("artifacts/c1_hi_small", batch_size=128, seed=42)
config = StageAConfig(feature_dim=len(manifest["feature_names"]), max_len=manifest["configuration"]["max_len"], seed=42)
model, history = train_stage_a(loaders, config, epochs=10)  # ~65 s en Apple Silicon (MPS)

device = resolve_device()
validation = score_loader(model, loaders["validation"], device)
threshold_info = select_threshold(validation["score"], validation["y"])
save_checkpoint("artifacts/checkpoints/stage_a.pt", model, config, threshold_info, feature_names=manifest["feature_names"])
```

El entrenamiento itera solo `loaders["train_normal"]`; la AUC-PR de validación se usa exclusivamente para elegir el mejor checkpoint por época, nunca para el gradiente. El umbral se elige maximizando F1 sobre la curva precisión-recall de validación, justificado por la prevalencia extrema (~0.72% positivos). Detalle completo, resultados y limitaciones honestas en [report/c2a_etapa_a.md](report/c2a_etapa_a.md) y en la sección C2A de [notebooks/proyecto2.ipynb](notebooks/proyecto2.ipynb).

## Correr el MVP

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Por defecto usa las copias congeladas en `app/data/c1_hi_small`, `app/model/stage_a.pt` y `app/model/stage_b.pt` (generadas con la misma semilla y configuración del pipeline principal), para que el MVP funcione sin depender de los CSV originales ni de reentrenar. La barra lateral permite apuntar a otras rutas de artefactos o checkpoint si se regeneran localmente; si no se encuentra el checkpoint de la Etapa B, el MVP sigue funcionando solo con la Etapa A.

