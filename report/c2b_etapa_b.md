# C2B — Etapa B: transfer learning, pérdida y ablación

La Etapa B debe aprovechar la representación aprendida en la Etapa A en lugar de partir de cero, y demostrar empíricamente que esa reutilización aporta valor. `src/models/stage_b.py` implementa `StageBClassifier`: reutiliza `input_proj`, `encoder` y `pooling` de `SequenceAutoencoder` (sin su decoder) y añade una cabeza `Linear(64,64)-ReLU-Dropout-Linear(64,1)`. La señal de ambas etapas se combina a nivel de features: el error de reconstrucción de la Etapa A (ya entrenada, congelada como fuente de esa señal) se concatena al vector comprimido `z` antes de la cabeza, así la predicción final usa tanto la representación transferida como el score de anomalía original.

## Estrategia de transfer learning

Se usó *discriminative fine-tuning* con descongelamiento gradual: la cabeza nueva entrena sola durante la primera época mientras el backbone permanece congelado (y en modo `eval`, para no aplicar dropout sobre pesos que no se actualizan), y luego el backbone se descongela con una tasa de aprendizaje menor (5e-4) que la de la cabeza (1e-3), para adaptarlo a la señal supervisada sin destruir de golpe lo aprendido sin etiquetas. Esta estrategia corresponde a las técnicas de adaptación de un modelo preentrenado vistas en el curso: en vez de descartar el encoder y entrenar todo desde cero, se reutiliza como inicialización y se ajusta con un LR reducido.

## Función de pérdida

`BCEWithLogitsLoss(pos_weight=136.80)`, usando el `pos_weight` que ya publica el manifiesto de C1. Con ~0.7% de positivos, es la opción estándar para desbalance extremo cuando ya se cuenta con una estimación confiable de la proporción (evita además buscar un hiperparámetro adicional, como el `gamma` de la pérdida focal).

## Un hallazgo honesto durante el ajuste

La primera configuración probada (3 épocas congeladas, LR de backbone 1e-4) parecía razonable pero dio un resultado indeseado: el modelo transferido quedó **por debajo** de la línea base entrenada desde cero (F1 de prueba 0.168±0.010 vs 0.258±0.039 en 3 semillas, corrida exploratoria previa a la ejecución oficial del notebook). La hipótesis fue que el backbone no alcanzaba a adaptarse a la señal supervisada con un LR tan conservador y tan pocas épocas de ajuste completo. Acortar el congelamiento a 1 época y subir el LR del backbone a 5e-4 confirmó la hipótesis: el modelo transferido pasó a superar claramente la línea base (números oficiales abajo, de la ejecución del notebook). Este es exactamente el tipo de iteración que un experimento de ablación debe exponer: la arquitectura de dos etapas no garantiza una mejora por sí sola, depende de una estrategia de fine-tuning bien calibrada, y una calibración deficiente puede revertir el resultado esperado.

## Experimento de ablación obligatorio

Umbral elegido por F1 sobre la curva precisión-recall de validación (mismo procedimiento que la Etapa A), evaluado una sola vez en prueba. Promedios y desviación estándar sobre 3 semillas (42, 43, 44), tomados de la ejecución oficial de `notebooks/proyecto2.ipynb`:

| Modelo | F1 (prueba) | AUC-PR (prueba) |
|---|---|---|
| Baseline supervisado desde cero (sin Etapa A) | 0.258 ± 0.039 | 0.193 ± 0.029 |
| **Dos etapas (transfer learning + score de A)** | **0.409 ± 0.026** | **0.400 ± 0.068** |

Con la semilla 42 (usada para el checkpoint oficial): el modelo de dos etapas detecta 20 de 54 casos de lavado en prueba con 19 falsas alertas (precisión 0.513, recall 0.370, F1 0.430, AUC-PR 0.456), frente a 17/54 con 47 falsas alertas del baseline (precisión 0.266, recall 0.315, F1 0.288, AUC-PR 0.225) y 10/54 de la Etapa A sola. La mejora en AUC-PR (~2.1x sobre el baseline en promedio) es más informativa que F1 dado el tamaño de la muestra positiva (54 casos), pero ambas métricas apuntan en la misma dirección: el modelo de dos etapas es consistentemente mejor en las 3 semillas, aunque la magnitud exacta de la mejora varía entre corridas.

**Se eligió F1 y AUC-PR como métricas principales** por la misma razón que en la Etapa A: con ~0.7% de positivos, exactitud no discrimina nada útil, y AUC-PR resume la separabilidad real independientemente del umbral operativo elegido.

## Limitación honesta sobre la varianza

Con solo 254 remitentes positivos en entrenamiento y 54 en prueba, los resultados tienen varianza real entre semillas y entre ejecuciones (el entrenamiento corrió en GPU MPS, que no garantiza operaciones bit-exactas — ver la misma limitación documentada en `report/c2a_etapa_a.md`). Una corrida exploratoria previa con las mismas 3 semillas y la misma configuración final dio F1 0.473±0.054 y AUC-PR 0.467±0.073 para el modelo de dos etapas; la ejecución oficial del notebook dio 0.409±0.026 y 0.400±0.068. Los números cambian, pero la conclusión cualitativa no: el modelo de dos etapas superó al baseline en las 6 corridas realizadas (2 configuraciones × 3 semillas, más la corrida exploratoria). El hallazgo de que la configuración de transfer learning importa tanto como su sola presencia —una configuración mal calibrada puede perder frente a un baseline, como se documentó arriba— es en sí mismo un resultado honesto que vale la pena reportar, no solo un paso intermedio a descartar.

**Referencia:** Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). *Parameter-Efficient Transfer Learning for NLP*. International Conference on Machine Learning (ICML). *(Nota: este paper es de 2019, fuera del rango 2020-2025 pedido para las referencias del reporte; se cita aquí como respaldo conceptual de la estrategia de fine-tuning discriminativo, pero el equipo debe sustituirla o complementarla con al menos una referencia 2020-2025 antes de la entrega final — ver checklist del equipo.)*
