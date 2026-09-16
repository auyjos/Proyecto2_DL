# CC3092 - Deep Learning

# Proyecto 2

## Instrucciones

- Esta es una actividad en grupos de no más de 3 integrantes.
  - Recuerden **unirse al grupo de canvas**
- No se permitirá ni se aceptará cualquier indicio de copia. De presentarse, se procederá según el reglamento correspondiente.
- Tendrán hasta el día indicado en Canvas.
  - No se confíen, aprovechen el tiempo en clase para entender todos los ejercicios y avanzar lo más posible.
- **NOTA:** Limiten el uso de IA generativa. Intenten primero buscar en fuentes de internet y si en verdad necesitan usarla, asegúrense de colcar el prompt que utilizan para cada task donde corresponda, así como una explicación de por qué ese prompt funcionó.

Guatemala recibe más de $20 mil millones anuales en remesas, equivalentes al 19% del PIB nacional. Ese volumen convierte al corredor de remesas Guatemala-Estados Unidos en uno de los más importantes del mundo y, simultáneamente, en uno de los más vulnerables a ser utilizado para lavado de dinero y financiamiento de actividades ilícitas. Las empresas que procesan remesas (Western Union, MoneyGram, Remitly) y los bancos que las reciben tienen obligaciones regulatorias de detectar patrones sospechosos bajo marcos como la Ley Contra el Lavado de Dinero u Otros Activos de Guatemala y las regulaciones FinCEN en Estados Unidos.

El problema técnico central es que los sistemas de detección actuales basados en reglas tienen tasas de falsos positivos del 95-99%, lo que significa que los equipos de cumplimiento deben revisar manualmente cientos de alertas para encontrar un caso real. Eso es insostenible a escala y deja espacio para que patrones sofisticados de lavado pasen desapercibidos.

Este proyecto propone construir un sistema de detección inteligente que aprenda qué es un patrón normal de remesas antes de aprender qué es sospechoso, produciendo alertas con trazabilidad: el sistema debe poder explicar qué transacciones específicas de un remitente activaron la alerta..

## Datos

**Dataset principal:** PaySim - Synthetic Financial Dataset for Fraud Detection, disponible en Kaggle. Generado por el Banco Mundial a partir de datos reales de un servicio de dinero móvil africano, con 6.3 millones de transacciones financieras sintéticas que replican patrones reales de uso y de fraude.

**Dataset complementario:** IBM AML Anti Money Laundering, disponible en Kaggle. Diseñado específicamente para investigación en detección de lavado de dinero con patrones etiquetados por expertos en cumplimiento.

Acceso:

```text
# PaySim
https://www.kaggle.com/datasets/ealaxi/paysim1

# IBM AML
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
```

**Instrucción importante:** los datos son masivos. El grupo debe tomar decisiones explícitas y justificadas sobre qué subconjunto usar, cómo construir las secuencias por remitente, y cómo balancear el dataset para el entrenamiento. Esas decisiones deben documentarse en el reporte con justificación técnica, no simplemente ejecutarse sin explicación

## Estructura del Proyecto

El proyecto tiene cuatro componentes con pesos distintos. Los componentes 1 y 2 son técnicos y se entregan como notebook ejecutado. El componente 3 es analítico y se entrega como reporte escrito. El componente 4 es el MVP funcional.

### Componente 1: Ingeniería de datos y representación de secuencias

El primer desafío del proyecto no es modelar sino representar. Una transacción aislada contiene poca información sobre la intención del remitente. Lo que revela lavado de dinero es el patrón a lo largo del tiempo: la frecuencia inusual, los montos que se fragmentan justo debajo de umbrales regulatorios, los destinos que cambian abruptamente, la concentración en horarios específicos.

El grupo debe construir un pipeline que transforme las transacciones individuales en secuencias por remitente, donde cada secuencia capture el comportamiento histórico de ese remitente como una serie ordenada temporalmente.

Las decisiones que el grupo debe tomar y documentar:

- Qué features representan cada transacción dentro de la secuencia (monto, tipo, hora, destino, frecuencia, etc.) y cómo se normalizan
- Qué longitud máxima de secuencia usar y qué hacer con remitentes que tienen muy pocas transacciones
- Cómo manejar el desbalance extremo entre remitentes normales y sospechosos
- Cómo separar entrenamiento, validación y prueba sin filtración de datos entre splits

**Verificación:** el notebook debe mostrar explícitamente la distribución de longitudes de secuencia, la proporción de casos positivos y negativos, y al menos tres ejemplos visualizados de secuencias normales versus sospechosas antes de cualquier modelado.

### Componente 2: Sistema de detección en dos etapas

El núcleo técnico del proyecto. El grupo debe diseñar e implementar un sistema que aborde el problema desde dos perspectivas complementarias y combine sus señales en una predicción final.

#### Etapa A: Aprendizaje de la normalidad

Antes de intentar clasificar qué es sospechoso, el sistema debe aprender qué es normal. La idea es que un modelo entrenado exclusivamente sobre comportamiento normal de remesas aprenderá una representación comprimida de ese comportamiento. Cuando se le presente una secuencia que no se parece a nada que haya visto, fallará en reproducirla correctamente. Esa dificultad de reproducción es la señal de anomalía.

El grupo debe seleccionar e implementar una arquitectura apropiada para este objetivo. La arquitectura debe ser capaz de procesar secuencias de longitud variable, producir una representación comprimida del comportamiento del remitente, y reconstruir la secuencia original desde esa representación. El error de reconstrucción por secuencia se convierte en el score de anomalía.

Una vez entrenado el modelo, el grupo debe definir un umbral sobre el error de reconstrucción que separe comportamiento normal de anómalo. Esa decisión no es arbitraria: debe justificarse con una métrica de evaluación apropiada sobre el conjunto de validación.

#### Etapa B: Aprendizaje supervisado sobre casos etiquetados

Con el conocimiento de representación aprendido en la Etapa A, el grupo debe construir un clasificador que aprenda a distinguir lavado de dinero de comportamiento legítimo usando los casos etiquetados disponibles. El clasificador debe aprovechar lo aprendido en la Etapa A en lugar de aprender desde cero, aplicando alguna de las estrategias de transfer learning estudiadas en la Semana 9.

El grupo debe justificar explícitamente qué estrategia de adaptación eligió y por qué, qué función de pérdida es apropiada dado el desbalance de clases y la naturaleza del problema, y cómo combina las señales de ambas etapas en una predicción final.

#### Experimento de demostración obligatorio:

El grupo debe demostrar empíricamente que la arquitectura de dos etapas aporta valor sobre una línea base de clasificador supervisado entrenado desde cero sin la Etapa A. Sin ese experimento, el proyecto no está técnicamente completo. Los resultados de demostración deben presentarse en una tabla comparativa con la métrica o métricas que el grupo considere más apropiadas para este problema, con justificación de esa elección.

### Componente 3: Análisis, interpretabilidad y reporte

Un sistema de detección de lavado de dinero que no puede explicar sus decisiones no puede usarse en un entorno regulatorio real. Los equipos de cumplimiento necesitan saber por qué una alerta fue generada antes de iniciar una investigación. Este componente evalúa la profundidad del análisis y la calidad del reporte.

#### Análisis de interpretabilidad

El grupo debe analizar qué transacciones específicas de una secuencia activaron la alerta del sistema. Esto implica examinar los pesos de atención del modelo sobre la secuencia y conectar los patrones encontrados con tipologías conocidas de lavado de dinero en remesas. El grupo debe seleccionar al menos cinco casos del conjunto de prueba, tres correctamente detectados y dos falsos positivos o falsos negativos, y analizar en profundidad qué hizo el modelo en cada caso y por qué.

#### Reporte ejecutivo

El reporte debe tener entre 2,000 y 3,000 palabras y cubrir:

- El problema de negocio en contexto regulatorio guatemalteco y centroamericano, con referencias a la legislación AML vigente
- Las decisiones de diseño del sistema con justificación técnica y de negocio
- Los resultados del experimento de ablación con análisis de qué aporta cada etapa
- El análisis de casos específicos detectados y no detectados
- Las limitaciones del sistema y qué se necesitaría para llevarlo a producción en una empresa de remesas real
- Al menos tres referencias a papers de venues indexadas (NeurIPS, ICML, ICLR, ACL, o journals de compliance financiero) publicados entre 2020 y 2025

El reporte debe poder ser leído por un director de cumplimiento de un banco guatemalteco que no es experto en deep learning. Eso significa que los conceptos técnicos deben explicarse con lenguaje de negocio y los resultados deben traducirse a implicaciones operativas concretas.

### Componente 4: MVP funcional

El grupo debe construir una interfaz funcional que demuestre el sistema en uso. La interfaz puede ser un artifact de Claude, una aplicación Streamlit, o cualquier herramienta que produzca una experiencia interactiva sin requerir instalación del evaluador.

La interfaz debe permitir:

- Ingresar o seleccionar un remitente del conjunto de prueba
- Visualizar su secuencia de transacciones con información básica de cada una
- Ver el score de anomalía de la Etapa A y la probabilidad de lavado de la Etapa B
- Ver un mapa de calor sobre la secuencia que muestre qué transacciones contribuyeron más a la alerta
- Generar automáticamente un párrafo de explicación en lenguaje natural del por qué de la alerta

El MVP no necesita ser visualmente elaborado pero debe funcionar sin errores en el momento de la presentación. Un MVP que no corre vale cero puntos independientemente de la calidad del resto del proyecto.

## Entregables

Tres archivos subidos a la plataforma antes del **domingo 20 de septiembre a las 11:59 PM**:

1. **Notebook ejecutado (.ipynb):** con todas las celdas con salida visible, código comentado, y los experimentos de ablación claramente identificados. El notebook debe poder reproducirse desde cero en menos de 30 minutos en Google Colab con GPU T4 gratuita.
2. **Reporte (.pdf):** el documento ejecutivo descrito en el Componente 3, con las referencias en formato APA o IEEE.
3. **Enlace al MVP:** URL de la interfaz funcional en un archivo de texto (.txt). Si es un artifact de Claude debe ser público compartido en enlance al mismo. Si es Streamlit debe estar desplegado en Streamlit Cloud (gratuito).

## Evaluación

| Componente | Criterio | Pts |
|---|---|---:|
| **C1: Ingeniería de datos** | Secuencias bien construidas, decisiones documentadas, visualizaciones | 20 |
| **C2A: Etapa A** | Arquitectura apropiada, entrenamiento solo sobre normalidad, umbral justificado | 20 |
| **C2B: Etapa B** | Transfer learning desde Etapa A, pérdida justificada, ablation completo | 20 |
| **C3: Análisis** | Interpretabilidad de atención, análisis de casos, reporte ejecutivo | 25 |
| **C4: MVP** | Funciona sin errores, muestra ambas etapas, mapa de calor, explicación en lenguaje natural | 15 |
| **Total** |  | **100** |

### Criterios de evaluación transversales

Estos criterios aplican a todos los componentes y pueden sumar o restar puntos del total:

**Justificación de decisiones:** cada decisión de diseño (arquitectura, función de pérdida, umbral, estrategia de transfer learning, métricas) debe tener una justificación explícita. Una decisión sin justificación se evalúa como si fuera incorrecta aunque el resultado sea bueno.

**Conexión con el curso:** el reporte y el notebook deben conectar explícitamente las técnicas usadas con las semanas del curso donde se desarrollaron. No basta implementar: el grupo debe demostrar que entiende por qué esa técnica es apropiada para este problema en términos de lo que aprendió.

**Honestidad sobre limitaciones:** un proyecto que reporta honestamente qué no funcionó y por qué es más valioso que uno que solo reporta lo que salió bien. El análisis de fallos es parte de la ingeniería.

**Uso de IA generativa:** el uso de herramientas como Claude, ChatGPT o Gemini está permitido y es esperado. Sin embargo, el grupo debe incluir al final del reporte una sección de máximo 200 palabras describiendo cómo usaron esas herramientas y qué decisiones tomaron ellos mismos. Un proyecto donde la IA tomó todas las decisiones de diseño sin criterio propio del grupo se penaliza en los criterios de justificación.

## Preguntas frecuentes anticipadas

**¿Pueden usar librerías de alto nivel como HuggingFace Transformers?** Para el componente de NLP pueden usar modelos preentrenados de HuggingFace. Para la arquitectura central del sistema de detección deben implementar los componentes clave con PyTorch, no simplemente llamar una función de alto nivel que resuelva el problema entero.

**¿Qué pasa si el dataset es demasiado grande para Google Colab?** Usar un subconjunto justificado es completamente válido. Lo que no es válido es usar un subconjunto sin justificación o usar tan pocos datos que el modelo no aprenda nada significativo.

**¿El MVP tiene que estar desplegado en internet?** Sí. Un notebook local no cuenta como MVP. Si tienen problemas con Streamlit Cloud, un artifact de Claude con los pesos del modelo embebidos como JSON es una alternativa válida.

**¿Pueden usar el mismo modelo para las dos etapas?** Pueden compartir componentes entre etapas pero las dos etapas deben ser distinguibles: una que aprende sin etiquetas y otra que aprende con etiquetas. Un modelo que solo hace clasificación supervisada desde el inicio no cumple con el diseño del proyecto.

---

# División de trabajo y estado actual

> Esta sección es una propuesta de organización del equipo y **no forma parte del PDF original**.

**Actualizado:** 11 de septiembre de 2026, a partir de los entregables verificables en la rama `feature/c1-sequence-data-pipeline`.

## Integrantes

- **José Auyón**
- **Jose Ruiz**
- **Gerardo Fernandez**

## Asignación principal

| Integrante | Ownership principal | Responsabilidades concretas |
|---|---|---|
| **José Auyón** | **Componente 1: Ingeniería de datos y representación de secuencias** | C1 implementado: selección y justificación de HI-Small; validación y feature engineering; secuencias por remitente; normalización; longitud máxima de 64; prevalencia natural; splits por `Entity ID`; visualizaciones; artefactos, DataLoaders y recuperación de remitentes para los demás componentes. Pendiente: apoyar la integración final y verificar el notebook completo en Colab. |
| **Jose Ruiz** | **Componente 2A (completo) + base del MVP + secciones de negocio del reporte** | C2A implementado, entrenado y evaluado (`report/c2a_etapa_a.md`). MVP en Streamlit con la Etapa A integrada de punta a punta, con `src/inference.run_stage_b` como contrato de extensión ya cerrado para que Gerardo lo conecte sin ayuda de Ruiz. Deployment del MVP actual en Streamlit Cloud. Contexto regulatorio y limitaciones/producción del reporte (`report/contexto_regulatorio.md`, `report/limitaciones_y_produccion.md`). |
| **Gerardo Fernandez** | **Componente 2B + interpretabilidad + integración final** | Construir el clasificador supervisado reutilizando lo aprendido en Etapa A; decidir estrategia de fine-tuning/transfer learning; justificar la función de pérdida para el desbalance; combinar las señales A y B; implementar la línea base supervisada desde cero; ejecutar la comparación/ablación; extraer pesos de atención; generar información para el heatmap; seleccionar y analizar los 5 casos requeridos; **conectar la Etapa B en `src/inference.run_stage_b`** (el contrato `StageBResult` ya está definido, no requiere cambios de interfaz); **ensamblar el reporte final** (compilar las secciones de los tres, referencias, sección de uso de IA) por ser quien genera los últimos resultados. |

**Por qué se reorganizó así (14 sept):** en la versión anterior de este documento, Ruiz integraba el MVP *después* de que Gerardo terminara C2B, lo que obligaba a Ruiz a volver a trabajar en su propia parte una vez que Gerardo avanzara. Con el contrato de `src/inference.run_stage_b` ya cerrado y probado (`tests/test_inference.py`, `streamlit.testing.v1.AppTest`), Ruiz puede dar por completada su parte hoy sin esperar a Gerardo, y es Gerardo —que de todas formas necesita ejecutar último para tener resultados que ablacionar— quien conecta su propio modelo al punto de extensión y arma el reporte final. Ningún integrante debe volver a una tarea que ya cerró.

## Estado verificable en esta rama

| Frente | Responsable | Estado | Evidencia disponible |
|---|---|---|---|
| C1: datos y secuencias | José Auyón | **Completado** | `src/data/sequences.py`, `tests/test_sequences.py`, `notebooks/proyecto2.ipynb`, contrato, reporte y gráficas. |
| C2A: aprendizaje de normalidad | Jose Ruiz | **Completado en `feature/c2a-stage-a-mvp`** | `src/models/stage_a.py`, `tests/test_stage_a.py`, sección C2A ejecutada en `notebooks/proyecto2.ipynb`, checkpoint en `artifacts/checkpoints/stage_a.pt`, resultados en `report/c2a_etapa_a.md`. |
| C2B: clasificación y ablación | Gerardo Fernandez | **Borrador completo, pendiente de revisión de Gerardo** | `src/models/stage_b.py`, `tests/test_stage_b.py`, sección C2B ejecutada en `notebooks/proyecto2.ipynb`, checkpoints en `artifacts/checkpoints/stage_b_*.pt`, resultados en `report/c2b_etapa_b.md` y `report/interpretabilidad_casos.md`. Gerardo debe revisar las decisiones (estrategia de fine-tuning, loss, arquitectura de la cabeza) y quedárselas o cambiarlas con criterio propio antes de la entrega. |
| C3: reporte e interpretabilidad | Equipo | **Borrador casi completo** | José, Ruiz y el borrador de Etapa B entregaron sus secciones; falta consolidación grupal, verificación de Gerardo, y que el equipo complete la referencia 2020-2025 que falta (ver `report/c2b_etapa_b.md`). |
| C4: MVP | Jose Ruiz, con interfaces del equipo | **Etapa A y borrador de Etapa B integrados** | `app/streamlit_app.py` funcional con ambas etapas (probado con `streamlit.testing.v1.AppTest`); falta que Gerardo revise/apruebe el modelo de la Etapa B y desplegar en Streamlit Cloud. |

## Trabajo compartido para el Componente 3

El **reporte no conviene dejarlo a una sola persona**, porque debe justificar decisiones tomadas en todos los componentes. La división recomendada es:

### José Auyón

- [x] Contexto del dataset y decisiones de ingeniería de datos.
- [x] Justificación del subconjunto elegido.
- [x] Construcción de secuencias, features, normalización, balance y splits.
- [x] Limitaciones de los datos sintéticos y de la evaluación.
- [ ] Apoyar la integración y revisión final del notebook completo cuando C2A/C2B estén disponibles.

### Jose Ruiz

- [x] Arquitectura y justificación de la Etapa A.
- [x] Entrenamiento sobre normalidad.
- [x] Elección del umbral y métrica de validación.
- [x] Descripción técnica y operativa del MVP.
- [x] Integración del score de anomalía en la interfaz.
- [x] Contexto de negocio y regulatorio (Guatemala/FinCEN).
- [x] Limitaciones del sistema y qué se necesita para producción.

Secciones escritas en `report/c2a_etapa_a.md`, `report/contexto_regulatorio.md` y `report/limitaciones_y_produccion.md`. Estas dos últimas no dependen de los resultados de Gerardo; solo necesitarán 1-2 frases de ajuste cuando exista la tabla de ablación (marcado con 🔶 en el archivo).

### Gerardo Fernandez

> **Nota (14 sept):** para no bloquear el avance del equipo, Jose Ruiz preparó un borrador completo de C2B (código, tests, notebook ejecutado, checkpoints y reporte) con ayuda de IA, dejando el registro de decisiones en `report/c2b_etapa_b.md`. Es un punto de partida, no un reemplazo del criterio de Gerardo: cada casilla debe marcarse solo después de que él la revise, la entienda y decida si la acepta, la ajusta o la rehace. El objetivo es que Gerardo llegue a revisar, no a implementar desde cero.

- [x] (borrador) Estrategia de transfer learning de la Etapa B — discriminative fine-tuning con descongelamiento gradual; ver justificación y el ajuste de hiperparámetros documentado en `report/c2b_etapa_b.md`.
- [x] (borrador) Función de pérdida y manejo del desbalance — `BCEWithLogitsLoss(pos_weight=136.80)`.
- [x] (borrador) Experimento de ablación y tabla comparativa — 3 semillas, dos etapas vs. baseline desde cero.
- [x] (borrador) Interpretabilidad, heatmaps y análisis de los 5 casos — `report/interpretabilidad_casos.md`.
- [x] (borrador) Explicación de cómo las señales de A y B se combinan en la predicción final — concatenación del score de A al vector `z` antes de la cabeza de clasificación.
- [ ] Ensamblar `report/report.md` final a partir de las secciones ya escritas por los tres (referencias en APA/IEEE — falta al menos una referencia 2020-2025 adicional, ver nota en `report/c2b_etapa_b.md` —, y sección de uso de IA ≤200 palabras).
- [ ] Revisar el borrador completo y decidir qué se queda, qué se ajusta y qué se rehace.

## División específica del MVP

- **José Auyón:** interfaz entregada mediante `get_sender`, que carga un remitente y devuelve su secuencia procesada, máscara, metadatos y transacciones originales en el mismo orden. *(Cerrado.)*
- **Jose Ruiz:** interfaz completa del MVP con la Etapa A (`app/streamlit_app.py`), integración de inferencia, visualización de transacciones, heatmap de atención, explicación en lenguaje natural, y deployment en Streamlit Cloud. *(Cerrado; no requiere que Gerardo termine para funcionar.)*
- **Gerardo Fernandez:** `src/inference.run_stage_b` ya está conectado a un checkpoint de la Etapa B (borrador, `app/model/stage_b.pt`) y el MVP lo consume automáticamente. Gerardo debe revisar ese modelo y, si lo reemplaza por el suyo propio, solo necesita guardar el nuevo checkpoint en la misma ruta — no requiere tocar `app/streamlit_app.py`.

## Dependencias y orden recomendado

1. **C1 está cerrado y versionado:** las secuencias, features, máscaras y particiones ya tienen un contrato estable.
2. **C2A y el MVP (con Etapa A) están cerrados:** checkpoint en `artifacts/checkpoints/stage_a.pt` (y copia congelada en `app/model/stage_a.pt`), listos para transfer learning y para producir alertas ya mismo.
3. **Gerardo completa C2B de forma independiente:** usa el checkpoint de la Etapa A para transfer learning, corre el experimento de ablación obligatorio contra un clasificador desde cero, y conecta su resultado en `src/inference.run_stage_b`. No necesita que Ruiz haga nada más.
4. **Gerardo ensambla el reporte final** una vez tiene sus resultados, incorporando las secciones ya cerradas de José y Ruiz.
5. Revisión de consistencia grupal del notebook completo (Auyón + Gerardo, dado que son quienes tocan las últimas celdas) antes de la entrega.

## Checklist por integrante

### José Auyón

- [x] Descargar y explorar PaySim / IBM AML según la decisión final del equipo
- [x] Definir y justificar subconjunto
- [x] Definir features por transacción
- [x] Normalizar features usando únicamente secuencias normales de entrenamiento para las variables numéricas
- [x] Construir secuencias por remitente ordenadas temporalmente
- [x] Definir longitud máxima y política para secuencias cortas
- [x] Diseñar balanceo
- [x] Crear train / validation / test agrupados por entidad y sin entidades compartidas
- [x] Graficar distribución de longitudes
- [x] Mostrar proporción positivos / negativos
- [x] Visualizar 3 ejemplos normales y 3 sospechosos
- [x] Dejar dataset, artefactos y DataLoaders reutilizables por Ruiz y Gerardo
- [x] Escribir la sección correspondiente del reporte
- [x] Documentar el contrato, los prompts de IA y la reproducción desde el CSV
- [ ] Verificar con el equipo el notebook completo en Colab T4 después de integrar C2A/C2B

### Jose Ruiz

- [x] Implementar Etapa A en PyTorch
- [x] Soportar secuencias de longitud variable
- [x] Entrenar exclusivamente sobre normalidad
- [x] Calcular reconstruction error por secuencia
- [x] Definir threshold con métrica justificada
- [x] Guardar checkpoint y representación/encoder reutilizable
- [x] Exponer anomaly score para inferencia
- [x] Crear interfaz del MVP
- [x] Integrar selección de remitente y visualización de transacciones
- [x] Escribir la sección correspondiente del reporte (Etapa A)
- [x] Escribir contexto de negocio y regulatorio
- [x] Escribir limitaciones y camino a producción
- [x] Desplegar públicamente el MVP — se publicó como Claude Artifact con los pesos de la Etapa A embebidos, alternativa explícitamente válida según el enunciado. Falta solo guardar la URL en el `.txt` del entregable final.
- [ ] Guardar la URL pública del MVP en un `.txt` (entregable 3)

La integración de la Etapa B en el MVP (`src/inference.run_stage_b`) se movió al checklist de Gerardo: el contrato ya está cerrado y probado desde el lado de Ruiz, así que no requiere que Ruiz vuelva a tocar el MVP.

### Gerardo Fernandez

Checklist técnico detallado. Todo lo marcado `(borrador)` existe en código real, probado y ejecutado, pero como punto de partida para que Gerardo lo revise — no como trabajo suyo ya validado.

- [x] (borrador) Implementar baseline supervisado desde cero — `train_stage_b(..., stage_a_model=None, use_stage_a_score=False)`
- [x] (borrador) Reutilizar representación de Etapa A — `StageBClassifier.load_backbone_from_stage_a`
- [x] (borrador) Elegir y justificar estrategia de transfer learning — discriminative fine-tuning, ver `report/c2b_etapa_b.md`
- [x] (borrador) Elegir y justificar loss para desbalance — `BCEWithLogitsLoss(pos_weight=136.80)`
- [x] (borrador) Entrenar clasificador Etapa B — `artifacts/checkpoints/stage_b_transferred.pt`
- [x] (borrador) Definir combinación de señales A + B — score de A concatenado a `z` antes de la cabeza
- [x] (borrador) Ejecutar ablation / comparación obligatoria — 3 semillas, `report/c2b_etapa_b.md`
- [x] (borrador) Crear tabla de métricas — sección C2B del notebook
- [x] (borrador) Extraer pesos/contribuciones por transacción — atención del pooling de la Etapa B
- [x] (borrador) Preparar datos del heatmap — ya consumido por `app/streamlit_app.py`
- [x] (borrador) Analizar 3 casos correctamente detectados — `report/interpretabilidad_casos.md`
- [x] (borrador) Analizar 2 falsos positivos o falsos negativos — `report/interpretabilidad_casos.md` (1 FP + 1 FN)
- [x] (borrador) Escribir la sección correspondiente del reporte (Etapa B + interpretabilidad)
- [x] (borrador) Conectar `src/inference.run_stage_b` con el modelo entrenado
- [ ] **Revisar todo lo anterior y decidir qué se queda, qué se ajusta y qué se rehace**
- [ ] Ensamblar `report/report.md` final a partir de `report/c1_ingenieria_datos.md`, `report/c2a_etapa_a.md`, `report/contexto_regulatorio.md`, `report/limitaciones_y_produccion.md`, `report/c2b_etapa_b.md` e `report/interpretabilidad_casos.md`; agregar al menos una referencia 2020-2025 adicional y la sección de uso de IA (≤200 palabras, describiendo honestamente qué hizo la IA y qué decidió el equipo)

## Checklist final del equipo

- [ ] Notebook con todas las celdas ejecutadas y outputs visibles
- [ ] Código comentado
- [ ] Experimento de ablación claramente identificado
- [ ] Reproducible desde cero en menos de 30 minutos en Colab con T4 gratuita
- [ ] Reporte entre 2,000 y 3,000 palabras
- [ ] Contexto regulatorio Guatemala/Centroamérica
- [ ] Al menos 3 papers 2020-2025 de los venues/journals solicitados
- [ ] Referencias APA o IEEE
- [ ] Cinco casos de interpretabilidad analizados
- [ ] Limitaciones y pasos necesarios para producción
- [ ] Sección de uso de IA generativa de máximo 200 palabras
- [ ] Cada integrante guarda los prompts de IA que haya utilizado y explica por qué funcionaron donde corresponda
- [ ] MVP probado de punta a punta antes de presentar
- [ ] URL pública del MVP guardada en `.txt`
- [ ] Entrega antes del domingo 20 de septiembre a las 11:59 PM

## Recomendación de ownership en Git

Para evitar conflictos, una estructura simple podría ser:

```text
project-2/
├── notebooks/
│   └── proyecto2.ipynb
├── src/
│   ├── data/
│   │   └── sequences.py          # José Auyón
│   ├── models/
│   │   ├── stage_a.py            # Jose Ruiz
│   │   └── stage_b.py            # Gerardo Fernandez
│   ├── evaluation/
│   │   └── ablation.py           # Gerardo Fernandez
│   └── inference.py              # Integración de los tres
├── app/
│   └── streamlit_app.py          # Jose Ruiz, con interfaces de José y Gerardo
├── report/
│   └── report.md                 # Escrito por los tres
└── README.md
```

La regla más importante para que esto funcione es que **José Auyón cierre temprano el contrato de datos**: shape de las secuencias, nombres/orden de features, máscaras/padding, labels y splits. Si eso cambia tarde, rompe tanto la Etapa A como la Etapa B y retrasa el MVP.
