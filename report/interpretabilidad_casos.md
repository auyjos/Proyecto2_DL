# Interpretabilidad: análisis de 5 casos del conjunto de prueba

Los 5 casos se seleccionaron del conjunto de prueba con el modelo de dos etapas oficial (`artifacts/checkpoints/stage_b_transferred.pt`, semilla 42): tres detectados correctamente (verdaderos positivos), un falso positivo y un falso negativo, elegidos aleatoriamente (semilla 42) dentro de cada categoría. Los pesos de atención son los del pooling de la Etapa B (heredado de la Etapa A y ajustado con la señal supervisada), alineados con `transaction_ids` en el mismo orden que usa el MVP para el mapa de calor.

## Caso 1 (verdadero positivo) — remitente `["018617","803D6C480"]`, 6 transacciones

Score de la Etapa A: 0.1608 (por debajo del umbral 0.1722 — la Etapa A sola no lo habría marcado). Probabilidad de la Etapa B: 0.9719 (umbral 0.9583).

Tres operaciones ACH ocurren en el mismo instante (2022/09/08 22:16): 13,092.44 USD hacia la propia cuenta, 9,208.13 Yenes y 1,370,734.89 Yenes hacia un destinatario nuevo (0110057) — esta última es la transacción de lavado. La atención más alta no cae exactamente sobre esa operación (0.081), sino sobre las dos que la acompañan en el mismo instante ([3]=0.236 hacia el mismo banco nuevo, y [5]=0.263, una ACH posterior). Esto es una limitación honesta que vale la pena señalar: el modelo identifica correctamente el **grupo** de operaciones simultáneas como el evento anómalo (consistente con **estructuración/fragmentación** — dividir un monto grande en varias transferencias simultáneas), pero no siempre concentra el peso máximo exactamente en la fila etiquetada dentro de ese grupo.

## Caso 2 (verdadero positivo) — remitente `["001267","8030004E0"]`, 27 transacciones

Score de la Etapa A: 0.1586 (por debajo del umbral). Probabilidad de la Etapa B: 0.9704.

Este es el caso más extremo de los tres: la secuencia alterna una operación de reinversión hacia la propia cuenta (en Yuanes) con una transferencia ACH externa hacia un banco distinto en una moneda distinta — y las **13 transferencias externas son, sin excepción, operaciones de lavado etiquetadas**, dispersas entre Dólares, Rublos, Riales sauditas, Euros, Pesos mexicanos, Dólares australianos y más. Es un ejemplo textual de **layering**: dispersar fondos a través de muchas contrapartes, monedas y bancos para dificultar el rastreo. Aquí la atención es casi uniforme (0.02–0.06 en las 27 posiciones) en vez de concentrarse en una operación puntual: cuando la mitad del historial de un remitente es ilícito, no existe un contraste claro entre "lo normal de este remitente" y "lo anómalo", porque casi todo el historial se aparta del patrón poblacional aprendido. El modelo aun así clasifica correctamente (probabilidad 0.97) porque la secuencia completa —no una sola fila— se aleja de la normalidad.

## Caso 3 (verdadero positivo) — remitente `["0217959","8078A4130"]`, 33 transacciones

Score de la Etapa A: 0.1084 (muy por debajo del umbral — la Etapa A sola no lo detecta). Probabilidad de la Etapa B: 0.9654.

El historial es repetitivo (pares recurrentes de Credit Card/Cheque en Euros) salvo por un grupo de tres operaciones ACH simultáneas el 4 de septiembre: 9,643.99 Euros hacia la propia cuenta, 5,635.99 Yenes y 1,185,454.91 Yenes hacia un banco nuevo (0210690) — esta última es la operación de lavado. Igual que en el Caso 1, la atención se concentra en ese grupo ([10]=0.128, [11]=0.162) pero el pico exacto ([11]) recae en la operación acompañante, no en la etiquetada ([12]=0.078). El patrón se repite: un cambio de moneda hacia un destinatario nuevo, camuflado entre docenas de operaciones legítimas de bajo monto — **layering** clásico, con la misma limitación de precisión fila-por-fila ya señalada.

## Caso 4 (falso positivo) — remitente `["0243947","81094C5C0"]`, 4 transacciones, y=0

Score de la Etapa A: 0.0234 (muy por debajo del umbral). Probabilidad de la Etapa B: 0.9653 — apenas por encima del umbral (0.9583), un caso límite real.

Con solo 4 operaciones, no hay un patrón repetitivo claro contra el cual medir "lo normal" de este remitente: una reinversión grande (96,011.31 Shekels) y tres ACH de montos moderados hacia dos bancos distintos reciben atención relativamente pareja (0.18–0.36). Ninguna operación individual es evidencia fuerte de lavado, pero la combinación de un historial corto y montos que varían bastante entre sí basta para cruzar un umbral calibrado de forma conservadora. Es la misma limitación estructural de siempre: el sistema mide desviación del patrón individual, y con poco historial esa medida es más ruidosa.

## Caso 5 (falso negativo) — remitente `["002454","8016EFD70"]`, 4 transacciones, y=1

Score de la Etapa A: 0.0968 (por debajo del umbral). Probabilidad de la Etapa B: 0.6144 — muy por debajo del umbral (0.9583); el modelo "dudó" pero lejos de cruzar el punto de corte.

La operación de lavado real ([3], 5,500.55 USD por ACH hacia un banco nuevo) ocurre el 12 de septiembre, casi dos semanas después de las otras tres operaciones (todas del 1 de septiembre) y recibe la tercera atención más alta (0.222). La atención más alta (0.415) recae en cambio sobre la primera operación de la secuencia, una ACH hacia otro banco externo que **no** está etiquetada como lavado. El modelo trata "transferencia externa temprana en un historial corto" como señal de atención en general, sin distinguir con precisión cuál transferencia externa es la ilícita. Con un umbral calibrado para maximizar F1 dada una prevalencia de ~0.7%, una probabilidad de confianza moderada como 0.61 queda del lado conservador y no genera alerta.

## Conclusión

Los tres verdaderos positivos comparten un patrón: un grupo de transacciones que rompe la escala de montos, la moneda o el destinatario habitual del remitente. El mecanismo de atención identifica consistentemente ese grupo —incluso cuando la Etapa A por sí sola no lo detecta (casos 1 y 3)— pero no siempre distingue con precisión la fila individual etiquetada dentro del grupo, y en el caso más extremo (Caso 2, donde casi la mitad del historial es ilícito) la atención se diluye de forma casi uniforme porque no hay un "normal" claro contra el cual contrastar. El falso positivo y el falso negativo comparten la misma raíz: historiales muy cortos (4 transacciones) dan poca base para distinguir una desviación genuinamente ilícita de una simplemente inusual. Ningún resultado aquí debe leerse como una alerta accionable sin revisión humana — esa es precisamente la función del mapa de calor y la explicación en lenguaje natural del MVP.
