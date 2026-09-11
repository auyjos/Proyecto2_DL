# CC3092 Proyecto 2 — detección de lavado en secuencias

Este repositorio contiene la ingeniería de datos de C1 y el análisis que justifica usar IBM AML HI-Small para los modelos secuenciales. Los CSV originales y los artefactos generados se excluyen de Git por su tamaño.

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

La explicación completa del contrato está en [docs/contrato_datos_c1.md](docs/contrato_datos_c1.md) y la sección propuesta para el reporte en [report/c1_ingenieria_datos.md](report/c1_ingenieria_datos.md).
