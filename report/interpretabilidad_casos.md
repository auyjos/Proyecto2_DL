# Interpretabilidad: análisis de 5 casos del conjunto de prueba

Los 5 casos se seleccionaron del conjunto de prueba con el modelo de dos etapas oficial (`artifacts/checkpoints/stage_b_transferred.pt`, semilla 42): tres detectados correctamente (verdaderos positivos), un falso positivo y un falso negativo, elegidos aleatoriamente (semilla 42) dentro de cada categoría. Los pesos de atención son los del pooling de la Etapa B (heredado de la Etapa A y ajustado con la señal supervisada), alineados con `transaction_ids` en el mismo orden que usa el MVP para el mapa de calor.

## Caso 1 (verdadero positivo) — remitente `["0217959","8078A4130"]`, 33 transacciones

Score de la Etapa A: 0.1084 (**por debajo** del umbral 0.1722 — la Etapa A sola habría dejado pasar este caso). Probabilidad de la Etapa B: 0.9917 (umbral 0.9797).

El historial es muy repetitivo: pares recurrentes de Credit Card/Cheque en Euros hacia dos destinatarios fijos, varias veces por semana. En medio de ese patrón aparece, el 4 de septiembre, una operación ACH de 1,185,454.91 Yenes hacia un banco nuevo (0210690) — la transacción de lavado real — precedida por otra ACH de 5,635.99 Yenes al mismo destino en el mismo instante. Esas dos transacciones concentran la atención más alta de toda la secuencia (0.135 y 0.126, frente a un promedio de ~0.02 en el resto). El patrón coincide con una tipología clásica de **layering**: mover fondos a través de un destinatario nuevo, en una moneda distinta a la habitual del remitente, camuflado entre docenas de operaciones legítimas de bajo monto. Es también el ejemplo más claro de por qué la arquitectura de dos etapas aporta valor: la Etapa A, entrenada solo sobre normalidad, no encontró suficiente error de reconstrucción global en una secuencia de 33 operaciones mayormente normales; la Etapa B, con la señal supervisada, sí aprendió a pesar fuertemente la operación puntual anómala.

## Caso 2 (verdadero positivo) — remitente `["001124","8065C8A80"]`, 5 transacciones

Score de la Etapa A: 0.2061 (por encima del umbral — este caso sí lo detecta la Etapa A sola). Probabilidad de la Etapa B: 0.9863.

Tres transacciones ACH ocurren en el mismo minuto (06:29): 12,349.60 USD hacia la propia cuenta del remitente, 110.01 CAD y 16,182.82 CAD hacia un destinatario nuevo (0024922) — esta última es la operación de lavado. Las tres concentran la atención más alta (0.258, 0.243, 0.276). El patrón —una operación pequeña y una grande hacia el mismo destinatario nuevo, ejecutadas simultáneamente con una tercera en otra moneda— es consistente con **fragmentación/estructuración**: dividir un monto para dificultar su seguimiento, combinada con un cambio de moneda hacia un destino no visto antes.

## Caso 3 (verdadero positivo) — remitente `["0222268","808AE2130"]`, 4 transacciones

Score de la Etapa A: 0.1945 (por encima del umbral). Probabilidad de la Etapa B: 0.9896.

Con solo 4 operaciones, la secuencia escala rápidamente: de una reinversión de 83,997.24 GBP a una ACH de 1,831,181.62 Yenes hacia un banco con un código atípicamente corto ("015"), la operación de lavado real. Esa última operación y la ACH inmediatamente anterior (hacia otro destinatario nuevo) concentran la atención (0.282 y 0.289). Es un ejemplo de **transferencia cruzada de alto valor con cambio de moneda y contraparte nueva**, detectable incluso con un historial muy corto porque el salto de magnitud es extremo.

## Caso 4 (falso positivo) — remitente `["0243614","81017DCE0"]`, 22 transacciones, y=0

Score de la Etapa A: 0.1076 (por debajo del umbral — la Etapa A no lo habría marcado). Probabilidad de la Etapa B: 0.9857, apenas por encima del umbral (0.9797) — un caso límite.

El historial es extremadamente repetitivo: casi todas las 22 operaciones alternan 440.45/43.10 Shekels en Credit Card/Cheque hacia el mismo banco (0119). La única operación distinta —1,211.51 Shekels por ACH hacia un banco nuevo (0220)— concentra la atención más alta (0.214, muy por encima del resto). El modelo reaccionó correctamente a "esto es distinto de lo que este remitente hace siempre", pero esa señal por sí sola no distingue una transacción legítima poco frecuente (p. ej. pagar a un proveedor nuevo una sola vez) de una realmente sospechosa. Es la limitación central del sistema: detecta desviación del patrón individual, no intención ilícita, y por eso el resultado de la Etapa B debe alimentar una revisión humana, no una decisión automática.

## Caso 5 (falso negativo) — remitente `["0013959","8094423B0"]`, 3 transacciones, y=1

Score de la Etapa A: 0.0102 (muy por debajo del umbral — prácticamente indistinguible de un remitente normal para la Etapa A). Probabilidad de la Etapa B: 0.76, por debajo de su umbral (0.9797) — el modelo sí "dudó" pero no lo suficiente para cruzar un umbral calibrado de forma conservadora dado el desbalance.

La secuencia tiene solo 3 operaciones: una reinversión pequeña, una reinversión grande (20,452.08 EUR, la de mayor atención, 0.391) y la operación de lavado real —13,073.91 EUR por ACH hacia un banco nuevo (0.373 de atención, la segunda más alta—. El modelo sí identificó la reinversión grande como lo más atípico, pero no priorizó correctamente la operación hacia el destinatario externo nuevo, que es la señal más relevante para lavado (una reinversión hacia la propia cuenta es mucho menos indicativa de layering que una transferencia externa). Con solo 3 operaciones, hay muy poco historial para que el modelo aprenda qué es "normal" para este remitente específico, y el umbral de F1-óptimo, calibrado sobre apenas 54 positivos de validación, es necesariamente conservador. Esto ilustra dos limitaciones honestas del sistema: (1) remitentes con historiales muy cortos son inherentemente más difíciles de evaluar, y (2) el punto de operación elegido (F1-óptimo) sacrifica recall por precisión, lo que garantiza que casos de confianza moderada como este queden sin alerta.

## Conclusión

Los tres verdaderos positivos comparten un patrón: una o dos transacciones que rompen abruptamente la escala de montos, la moneda o el destinatario habitual del remitente, y el mecanismo de atención las identifica consistentemente incluso cuando la Etapa A por sí sola no las habría marcado (casos 1 y, en menor medida, ninguno de los tres depende exclusivamente de A). El falso positivo y el falso negativo comparten la misma raíz: el modelo mide desviación del patrón individual del remitente, no evidencia directa de intención ilícita, y con historiales muy cortos (3-4 operaciones) tiene poca base para distinguir ambas cosas. Ningún resultado aquí debe leerse como una alerta accionable sin revisión humana — esa es precisamente la función del mapa de calor y la explicación en lenguaje natural del MVP.
