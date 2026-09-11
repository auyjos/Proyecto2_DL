# Contrato de datos C1 para modelos y MVP

## Decisiones fijadas

- Fuente de modelado: IBM AML `HI-Small_Trans.csv`, acompañada por `HI-Small_accounts.csv`.
- Secuencia: cuenta emisora identificada por `(From Bank, Account origen)`.
- Unidad de split: `Entity ID`. Todas las cuentas conocidas de una entidad seleccionada permanecen en un único conjunto.
- Muestra: 50,000 cuentas seleccionadas uniformemente, sin consultar etiquetas, usando `random.Random(42).sample(sorted(sender_keys), 50000)`.
- Recorte: últimas 64 operaciones según `(Timestamp, transaction_id)`; `transaction_id` es el índice cero basado de la fila de datos original.
- Etiqueta de secuencia: uno si alguna transacción retenida tiene `Is Laundering=1`.
- Balance: prevalencia natural. No se aplica SMOTE ni sobremuestreo. El manifiesto publica `pos_weight` como dato para la etapa B.

## Features y orden

Las primeras nueve columnas son:

1. `log_amount_paid_z`: `log1p(Amount Paid)`, estandarizado.
2. `log_delta_hours_z`: horas desde la operación anterior, con cero para la primera; `log1p` y estandarización.
3. `log_previous_24h_z`: número de operaciones anteriores dentro de las últimas 24 horas; `log1p` y estandarización.
4. `hour_sin` y 5. `hour_cos`.
6. `weekday_sin` y 7. `weekday_cos`.
8. `cross_bank`.
9. `destination_seen_before`, calculado solo con posiciones anteriores.

Después aparecen one-hot de moneda de pago y formato de pago, cada grupo con `<UNK>`. Los vocabularios se ordenan y se ajustan con todo entrenamiento. Media y desviación de las tres features logarítmicas se ajustan exclusivamente con transacciones de remitentes normales de entrenamiento. Una desviación cero se reemplaza por uno.

Importes de monedas diferentes no se convierten ni se consideran equivalentes; la moneda queda como contexto categórico. Labels, nombres, identificadores, índices y campos futuros se excluyen de `x`.

## Archivos persistidos

| Archivo | Contenido |
|---|---|
| `manifest.json` | Versión, configuración, huellas SHA-256, conteos, tiempos, features y rutas. |
| `splits.json` | Sender IDs, Entity IDs, objetivos y conteos por split. |
| `preprocessor.json` | Media, escala y vocabularios ajustados. |
| `sequences.npz` | Features concatenadas, offsets, labels, longitudes originales, IDs y códigos de split. |
| `transactions.parquet` | Filas originales retenidas para trazabilidad y MVP. |

El código valida `contract_version=1` antes de cargar. Si cambia la fuente, muestra, longitud, features o unidad de split, se deben regenerar todos los artefactos y repetir la etapa A, la etapa B y la línea base.

## Batch

```text
x                float32 [B, 64, F]
mask             bool    [B, 64]
lengths          int64   [B]
y                float32 [B]
sender_id        list[str]
entity_id        list[str]
transaction_ids  int64   [B, 64]
```

El padding de `x` es cero y el de `transaction_ids` es `-1`. Los loaders de entrenamiento barajan con un generador de PyTorch inicializado con la semilla solicitada; validación y prueba conservan el orden persistido.

## Resultado real de referencia

Con los archivos locales registrados en `artifacts/c1_hi_small/manifest.json`:

- 5,078,345 transacciones y 496,999 cuentas emisoras en la fuente.
- 464,009 transacciones seleccionadas antes del recorte y 426,244 retenidas.
- 1,307 secuencias recortadas.
- 366 remitentes positivos antes del recorte y 362 después.
- Splits exactos de 35,000/7,500/7,500 remitentes, con 254/54/54 positivos.
- 33 features y `pos_weight=136.7953` en entrenamiento.

## Límites

Los datos son sintéticos y no son remesas guatemaltecas observadas. Un label cero indica ausencia de operaciones etiquetadas dentro del historial retenido, no una certificación de legitimidad. La división evalúa generalización a entidades no vistas, pero no es una prueba prospectiva por tiempo. Agrupar por entidad reduce una fuente de fuga; destinatarios, bancos y patrones globales pueden aparecer en varios conjuntos.
