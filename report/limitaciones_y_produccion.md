# Limitaciones del sistema y camino a producción

**Autor de esta sección: Jose Ruiz.** *(Nota: los puntos marcados con 🔶 deben completarse con los números finales de la Etapa B/ablación una vez Gerardo los tenga; el resto ya es válido con los resultados actuales de la Etapa A.)*

## Limitaciones honestas

**Los datos son sintéticos, no remesas guatemaltecas reales.** IBM AML HI-Small modela bancos, individuos y empresas en general; sus patrones de lavado se diseñaron para investigación, no se observaron en el corredor Guatemala–Estados Unidos. Un resultado bueno aquí es evidencia de que la arquitectura funciona, no una garantía de que funcionará igual sobre remesas reales, que tienen su propia distribución de montos, frecuencias y tipologías (fraccionamiento de remesas familiares, uso de múltiples remitentes hacia un mismo beneficiario, etc.).

**La Etapa A por sí sola es una señal débil.** Como se documenta en `report/c2a_etapa_a.md`, el autoencoder entrenado solo sobre normalidad detecta en prueba 7 de 54 casos de lavado (recall 13%) con precisión de 37% en sus alertas. Es una señal muy por encima del azar, pero un sistema que se quedara solo en la Etapa A dejaría pasar a la mayoría de los casos reales. 🔶 *La tabla de ablación de Gerardo debe mostrar cuánto mejora esto al combinar ambas etapas; si la mejora es marginal, ese también es un resultado honesto que hay que reportar.*

**La partición de datos mide generalización a entidades nuevas, no capacidad predictiva prospectiva en el tiempo.** Los splits de C1 agrupan por `Entity ID` para evitar que una cuenta hermana aparezca en dos conjuntos, pero no simulan el escenario real de producción, donde el modelo debe predecir sobre transacciones que ocurren *después* del período de entrenamiento. Bancos y destinatarios pueden repetirse entre splits.

**No hay evaluación de robustez adversarial.** El proyecto no probó qué tan fácil sería para un actor que conoce el sistema evadirlo (por ejemplo, fragmentando montos de forma que se parezcan más a la distribución de "normalidad" aprendida). Esto es estándar dejarlo fuera del alcance de un proyecto académico, pero es una limitación real para cualquier despliegue.

**El entrenamiento en GPU MPS (Apple Silicon) no es bit-exacto.** Dos corridas con la misma semilla dieron AUC-PR de validación de 0.120 y 0.136 respectively; la conclusión cualitativa se mantuvo, pero los números exactos de este reporte son de una corrida específica, no una constante determinista del sistema.

**Las features actuales son genéricas, no específicas de remesas.** No incluyen señales que un oficial de cumplimiento real consideraría de alto valor: relación entre remitente y beneficiario (KYC), si el destinatario es una entidad de alto riesgo, historial de PEP (personas expuestas políticamente), o grafos de red entre múltiples remitentes hacia el mismo beneficiario.

## Qué se necesitaría para producción

1. **Datos reales bajo gobernanza de datos apropiada.** Acceso a transacciones reales de remesas requeriría acuerdos de confidencialidad, anonimización/tokenización de identificadores personales, y cumplimiento con protección de datos tanto guatemalteca como la regulación aplicable en Estados Unidos, antes de que un modelo como este pueda entrenarse sobre ellos.
2. **Un flujo humano-en-el-circuito, no una decisión automática.** El output de este sistema (score + heatmap + explicación) debe alimentar la cola de trabajo de un analista de cumplimiento, no generar un SAR automáticamente. El diseño de trazabilidad de este proyecto existe justamente para que ese analista pueda decidir rápido.
3. **Monitoreo de drift y recalibración periódica del umbral.** El umbral de la Etapa A se fijó una vez sobre un conjunto de validación estático; en producción, el comportamiento "normal" de los remitentes cambia con el tiempo (estacionalidad, nuevas rutas de remesas, inflación), y el umbral necesitaría revisarse con una cadencia definida, no quedar fijo indefinidamente.
4. **Validación prospectiva en el tiempo**, entrenando con datos históricos y evaluando sobre un período posterior real, no solo sobre entidades no vistas dentro del mismo período.
5. **Enriquecimiento de features con datos de KYC/beneficiario** y, si el volumen lo justifica, señales de grafo entre remitentes y destinatarios frecuentes.
6. **Registro de auditoría persistente de cada alerta y su explicación**, para poder reconstruir ante el regulador por qué el sistema marcó (o no marcó) una operación específica, meses después del hecho.
7. 🔶 *Definir, junto con la tabla de ablación de Gerardo, a qual costo operativo (alertas por revisar vs. casos detectados) el equipo recomendaría operar el sistema en un despliegue real — esa es una decisión de negocio, no solo técnica, y debe justificarse con la curva precisión-recall combinada de ambas etapas.*
