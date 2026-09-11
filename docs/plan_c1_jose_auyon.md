# Plan de trabajo de José Auyón — Componente 1

Fecha: 9 de septiembre de 2026.

## Objetivo y acuerdos

Preparar datos reproducibles para las etapas A y B, la línea base y el MVP, con decisiones justificadas y sin filtración entre remitentes de entrenamiento, validación y prueba.

Acuerdos tomados con José:

- Explorar PaySim y documentar su limitación para construir historiales; usar IBM AML HI-Small para las secuencias de los modelos y el MVP.
- Entregar un notebook en español y un módulo reutilizable de datos.
- Cubrir C1, su contribución al reporte y la integración final del notebook. Ruiz conserva la responsabilidad de C2A y del MVP; Gerardo, la de C2B, ablación e interpretabilidad.

El uso de IBM para los modelos se explicará expresamente frente al enunciado, que llama a PaySim dataset principal. No se ha obtenido una validación del docente sobre ese ajuste. Fraude en PaySim y lavado en IBM son etiquetas distintas; no se mezclarán.

## Evidencia obtenida antes de implementar

Se leyeron los CSV completos con Python, sin modificarlos ni generar datasets procesados.

La comparación quedó reproducida y documentada en [el notebook ejecutado](../notebooks/comparacion_datasets.ipynb), con [código comentado](../scripts/compare_datasets.py) y [resultados exportados](../reports/data_comparison/resumen.md). Se amplió a las seis variantes IBM y sus catálogos Accounts; ver [método, ejecución e implicaciones](analisis_variantes_ibm.md). Estos artefactos son exploratorios; no constituyen todavía el pipeline completo de C1.

**Hallazgo de la ampliación:** Accounts vincula distintas cuentas con una misma entidad. Antes de implementar los splits, revisar la recomendación de agrupar por `Entity ID`; la separación por cuenta descrita abajo no garantiza entidades disjuntas. La ampliación no cambia automáticamente el contrato ni las particiones propuestas.

| Medición | PaySim | IBM HI-Small |
|---|---:|---:|
| Transacciones | 6,362,620 | 5,078,345 |
| Remitentes | 6,353,307 | 496,999 |
| Remitentes con una transacción | 6,344,009 | 152,754 |
| Remitentes con al menos cinco transacciones | 0 | 128,779 |
| Remitentes con alguna transacción positiva | 8,213 | 3,376 |
| Máximo de transacciones por remitente | 3 | 168,672 |

En IBM, remitente significa el par de cadenas `(From Bank, Account de origen)`. El percentil 95 de longitud es 46 y el 99 es 92. PaySim tiene 99.85% de remitentes con una sola operación, lo que limita el aprendizaje de patrones temporales.

Una selección uniforme de 50,000 remitentes de IBM, usando `random.Random(42).sample(sorted(sender_keys), 50000)`, produjo:

- 464,009 transacciones antes del recorte y 426,244 después de conservar las 64 últimas por remitente.
- 1,307 remitentes recortados: 2.614% de la muestra.
- 366 remitentes positivos antes del recorte y 362 después de recalcular la etiqueta sobre las operaciones conservadas.
- 15,443 secuencias de longitud uno.

Esto valida una configuración inicial manejable; todavía no demuestra tiempos de entrenamiento ni calidad predictiva.

## 1. Preparar el proyecto y reproducir la exploración

- [x] Crear `notebooks/proyecto2.ipynb` para ejecutar el análisis, mostrar resultados y explicar decisiones. Reservar secciones para los componentes de los compañeros; mantener la lógica reutilizable en `src/data/sequences.py`.
- [x] Preparar dependencias de Python: pandas, NumPy, scikit-learn, PyTorch, PyArrow, Matplotlib y pytest. Registrar las versiones verificadas al ejecutar el pipeline en local.
- [x] Configurar exclusiones de Git para datos originales, datos procesados, entornos y checkpoints. Versionar código, configuración y documentación.
- [x] Leer IBM en streaming fila por fila. Seleccionar identificadores y después recuperar sus operaciones; evitar usar Medium o Large para el pipeline de modelado.
- [x] Asignar nombres únicos a las dos columnas `Account` por su posición: origen y destino. Leer bancos y cuentas como cadenas para conservar ceros iniciales.
- [x] Crear un `transaction_id` estable basado en la posición de la fila original. Validar fechas, importes finitos no negativos, identificadores y etiquetas binarias; detener el proceso con un diagnóstico ante valores inválidos, sin eliminar filas silenciosamente.
- [x] Informar duplicados exactos sin borrarlos automáticamente: dos operaciones iguales pueden ser eventos distintos.
- [x] Mostrar dimensiones, fechas, clases y longitudes de ambos datasets. No utilizar archivos de patrones ni etiquetas como entradas del modelo.

**Entrega verificable:** exploración reproducible y explicación respaldada por cifras de por qué IBM sostiene el modelado secuencial.

## 2. Fijar muestra, secuencias y particiones

- [x] Usar la selección uniforme de 50,000 remitentes descrita arriba, sin consultar sus etiquetas para seleccionarlos. Guardar los identificadores elegidos y la semilla.
- [x] Ordenar cada historial por fecha y `transaction_id` para resolver empates de forma determinista. Conservar las últimas 64 operaciones y mantener las secuencias cortas, incluso de longitud uno.
- [x] Definir `y = 1` si alguna operación conservada tiene `Is Laundering = 1`; en otro caso, `y = 0`. Guardar las etiquetas por transacción exclusivamente para evaluación y visualización.
- [x] Registrar cuántas operaciones y etiquetas positivas se pierden por el recorte. Explicar que un remitente con `y = 0` no está certificado como legítimo fuera del historial observado.
- [x] Crear 35,000 secuencias de entrenamiento, 7,500 de validación y 7,500 de prueba, agrupando por `Entity ID` y aproximando simultáneamente los tamaños y la prevalencia con semilla 42. En la muestra real se alcanzaron exactamente los tres tamaños.
- [x] Guardar el manifiesto de particiones antes de ajustar cualquier transformación. Cada remitente y cada entidad pertenecen a una sola partición; todos los modelos utilizan exactamente el mismo manifiesto.

**Decisión de evaluación:** medir generalización a entidades no vistas. Esta partición no representa una evaluación prospectiva en el tiempo ni garantiza destinatarios disjuntos; documentar esos límites. No utilizar estadísticas agregadas entre particiones.

## 3. Construir features, normalización y balance

Calcular las features después del recorte, sobre el historial visible y sin consultar operaciones futuras:

| Orden | Feature | Tratamiento |
|---|---|---|
| 1 | Importe pagado | `log1p(Amount Paid)` |
| 2 | Horas desde la operación anterior | `log1p(delta_horas)`; cero en la primera |
| 3 | Número de operaciones anteriores en las últimas 24 horas | `log1p(conteo)`; cero sin historial |
| 4–5 | Hora del día, incluyendo minutos | Seno y coseno con período de 24 horas |
| 6–7 | Día de la semana | Seno y coseno con período de siete días |
| 8 | Banco de origen distinto al de destino | Indicador binario |
| 9 | Destinatario ya observado antes en la secuencia | Indicador binario; destino identificado por banco y cuenta |
| 10 en adelante | Moneda de pago y formato de pago | One-hot, en ese orden; vocabularios ordenados y categoría desconocida |

- [x] Ajustar media y desviación de las tres features logarítmicas únicamente con transacciones válidas de secuencias normales de entrenamiento. Usar escala uno si la desviación es cero. Guardar esos parámetros y reutilizarlos sin reajuste en los demás conjuntos y en inferencia.
- [x] Ajustar vocabularios categóricos únicamente con entrenamiento. Mantener sin estandarizar las features cíclicas, binarias y one-hot. Guardar nombres y orden completo de las `F` features.
- [x] No convertir importes de monedas diferentes como si fueran equivalentes: conservar la moneda como contexto y documentar que no hay conversión cambiaria.
- [x] Excluir de `X` etiquetas, identificadores crudos, índices de fila y estadísticas futuras. Los identificadores permanecen en metadatos para trazabilidad.
- [x] Mantener prevalencia natural en los tres conjuntos, sin SMOTE ni sobremuestreo inicial. Exponer `train_normal` para la etapa A y `train` completo para la etapa B y la línea base.
- [x] Entregar conteos de clase y `pos_weight = negativos_train / positivos_train`. Gerardo decide su uso en la pérdida; el pipeline no aplica una segunda corrección de desbalance.

**Entrega verificable:** preprocesador persistido, features finitas y separación clara entre entradas del modelo e información para evaluación.

## 4. Entregar el contrato que utilizará el equipo

Interfaces públicas del módulo:

```python
build_artifacts(csv_path, output_dir, *, n_senders=50000, max_len=64, seed=42) -> dict
make_dataloaders(artifact_dir, *, batch_size=128, seed=42) -> dict
get_sender(sender_id, artifact_dir, *, split="test") -> dict
```

`build_artifacts` devuelve el manifiesto de generación. `make_dataloaders` devuelve `train`, `train_normal`, `validation` y `test`; solo los loaders de entrenamiento barajan las secuencias.

Cada batch contiene:

| Campo | Forma/tipo | Significado |
|---|---|---|
| `x` | `float32 [B, 64, F]` | Features; padding a la derecha con ceros después de transformar |
| `mask` | `bool [B, 64]` | `True` en transacciones reales; `False` en padding |
| `lengths` | `int64 [B]` | Longitudes entre 1 y 64 |
| `y` | `float32 [B]` | Etiqueta de la secuencia |
| `sender_id` | Lista de cadenas | Identificador estable: JSON compacto del par banco/cuenta |
| `transaction_ids` | `int64 [B, 64]` | Filas originales; `-1` para padding |

- [x] Persistir arrays válidos concatenados y offsets en NPZ, transformaciones y manifiestos en JSON, y las filas originales conservadas en Parquet. Agregar versión del contrato, semilla, checksum del CSV, conteos y tiempos; validar la versión al cargar.
- [x] Hacer que `get_sender` devuelva los campos del batch para un solo remitente, junto con `transactions`: filas originales en exactamente el mismo orden, con fecha, importe, moneda, formato, destinatario y etiqueta para evaluación. Un ID desconocido o fuera del split solicitado produce un error descriptivo.
- [x] Entregar a Ruiz y Gerardo un ejemplo de batch, dimensiones reales, instrucciones de carga y la regla de máscara. La reconstrucción, atención y pooling deben ignorar padding.
- [x] Comprobar que cargar artefactos produce los mismos valores y orden que el pipeline original. El MVP utilizará los artefactos guardados y no reajustará transformaciones.

## 5. Notebook, reporte y cierre

- [x] Antes del modelado, mostrar distribución de longitudes antes/después del recorte, proporciones de clases por split y número de secuencias disponibles para `train_normal`.
- [x] Visualizar tres pares de ejemplos normales/sospechosos de entrenamiento: seis secuencias en total. Graficar importe contra tiempo, indicar moneda y marcar transacciones positivas; seleccionar ejemplos con varias operaciones y semilla fija.
- [x] Escribir aproximadamente 500–650 palabras para el reporte sobre dataset, selección, features, recorte, balance, splits y limitaciones. Coordinar el espacio restante con el equipo para cumplir 2,000–3,000 palabras.
- [x] Explicar que son datos sintéticos y no remesas guatemaltecas observadas. Verificar la fuente original de IBM; investigar el contexto normativo en fuentes oficiales al redactar la sección regulatoria grupal.
- [x] Registrar prompts de IA, finalidad, qué aportaron y decisiones tomadas personalmente; facilitar el resumen grupal de máximo 200 palabras.
- [x] Documentar la ejecución a partir del CSV y el código del proyecto. Compartir con el equipo los artefactos procesados para integración, sin sustituir la comprobación de reproducción desde el CSV.
- [x] Medir por separado validación/selección, recolección/recorte, transformación/persistencia y checksum en local. C1 tarda menos de cinco minutos.
- [ ] Comprobar finalmente el notebook completo con C2A/C2B en Colab T4: debe correr desde cero en menos de treinta minutos.
- [ ] Integrar las secciones de Ruiz y Gerardo conservando el manifiesto y el orden de features; ejecutar todo el notebook y revisar salidas, tiempos y consistencia con el reporte y el MVP.

## Pruebas y criterios de aceptación

Usar fixtures pequeños para comprobar comportamiento y una ejecución real para comprobar volumen e integración:

1. **Lectura e identidad:** cuentas con ceros iniciales, cuentas iguales en bancos distintos y las dos columnas `Account` se interpretan correctamente; valores inválidos fallan con diagnóstico.
2. **Orden y recorte:** filas desordenadas y fechas empatadas producen el mismo orden; si la única operación positiva queda fuera de las últimas 64, la etiqueta conservada es cero.
3. **Ausencia de filtración:** intersecciones vacías de remitentes y transacciones entre splits; modificar validación/prueba no altera parámetros de normalización ni vocabularios; `train_normal` contiene únicamente `y = 0`.
4. **Causalidad:** una operación posterior no cambia features de operaciones previas dentro del mismo historial retenido; conteos de 24 horas y destinatarios previos excluyen la operación actual.
5. **Casos límite:** longitudes 1, 64 y mayores de 64; categoría desconocida; desviación cero; padding y máscaras coherentes y sin NaN o infinito.
6. **Contrato:** guardar/cargar conserva valores, etiquetas e IDs; las filas del MVP coinciden con posiciones válidas del tensor; etapa B y baseline reciben los mismos splits.
7. **Reproducibilidad:** con el mismo CSV, algoritmo y semilla, reproducir la selección de 50,000 remitentes, 426,244 transacciones conservadas y 362 secuencias positivas. Una diferencia debe investigarse y registrarse antes de cambiar el contrato.
8. **Aceptación final:** seis visualizaciones de ejemplo, gráficas de distribución y clases, artefactos cargables por ambos compañeros y notebook completo ejecutado en Colab dentro del límite.

## Orden y fechas objetivo

| Fecha objetivo | Resultado |
|---|---|
| 9–10 de septiembre | Exploración reproducible y contrato comunicado a Ruiz y Gerardo |
| 11–12 de septiembre | Pipeline, splits, transformaciones, artefactos y pruebas de C1 |
| 13 de septiembre | Notebook de C1 ejecutado y primera sección del reporte |
| 14–17 de septiembre | Soporte a integración y comprobación de tiempos del notebook completo |
| 18–19 de septiembre | Revisión de consistencia, ejecución final y prueba del MVP con el equipo |
| 20 de septiembre | Margen para correcciones y entrega antes de las 23:59 |

Estas fechas son objetivos propuestos, no compromisos ya acordados con los compañeros. Si la evaluación de rendimiento exige cambiar la muestra o longitud, regenerar todos los artefactos, incrementar la versión del contrato y repetir los experimentos de ambos modelos con la misma configuración.
