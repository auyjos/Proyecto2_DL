# Registro de prompts de IA — Jose Ruiz

Registro de los prompts usados para implementar C2A (Etapa A), el MVP y el borrador de C2B (Etapa B). El reporte final resume el uso de IA del equipo completo en un máximo de 200 palabras.

## Planificación y alcance

**Prompt:** definir si la rama debía crearse desde `main` o desde otra base, y planificar cómo completar en una sesión la Etapa A y el MVP.

**Por qué funcionó:** la IA leyó `docs/CC3092_Proyecto2_1a1_y_division_trabajo.md`, confirmó el ownership ya asignado (Componente 2A + lead del MVP) y recomendó `feature/c2a-stage-a-mvp` desde `main`, consistente con el patrón ya usado para C1. Se aceptó sin cambios por ser la opción más simple y coherente con el resto del repositorio.

## Arquitectura y entrenamiento de la Etapa A

**Prompt:** implementar en PyTorch la arquitectura de la Etapa A sobre el contrato de C1, sin modificar `src/data/sequences.py`.

**Por qué funcionó:** se evaluó un autoencoder Transformer con pooling por atención frente a alternativas recurrentes; se eligió el Transformer porque su encoder es reutilizable por transfer learning en la Etapa B y su atención sirve como señal de interpretabilidad. Las decisiones de umbral (F1 sobre la curva precisión-recall, justificado por la prevalencia de 0.72%) y de métrica de selección de checkpoint (AUC-PR de validación, nunca el gradiente) se revisaron y confirmaron antes de correr el entrenamiento real sobre los datos.

## Corrección de un bug de compatibilidad (MPS)

Al entrenar sobre los artefactos reales de C1, `nn.TransformerEncoder` con máscara de padding falló en el backend MPS (`NotImplementedError`, ruta de nested tensors no soportada). Se identificó la causa (optimización interna de PyTorch no disponible en MPS) y se corrigió con `enable_nested_tensor=False`, sin alterar la arquitectura. Se verificó que los tests existentes siguieran pasando antes de continuar.

## Resultados y limitaciones de la Etapa A

Los números reportados (AUC-PR de validación 0.136, F1 de prueba 0.192, 7/54 casos detectados) provienen de una ejecución real del notebook, no de estimaciones. Se documentó explícitamente que la AUC-PR cae después de la época 2 y que el resultado de la Etapa A sola es insuficiente para producción, por ser la motivación directa del experimento de ablación de la Etapa B.

## MVP y contrato de extensión

**Prompt:** construir el MVP en Streamlit con la Etapa A integrada, dejando un punto de extensión para conectar la Etapa B sin modificar la interfaz.

**Por qué funcionó:** se definió `src/inference.run_stage_b` como contrato tipado (`StageBResult`) que degrada explícitamente a `None` cuando no hay checkpoint, en vez de simular un resultado. El MVP se probó con `streamlit.testing.v1.AppTest`, incluyendo un caso positivo real, antes de considerarlo terminado.

## Implementación de la Etapa B (borrador para revisión del equipo)

**Prompt:** implementar también la Etapa B (transfer learning, pérdida para el desbalance, baseline desde cero, ablación obligatoria, interpretabilidad de 5 casos) como borrador para que el resto del equipo lo revise.

**Por qué funcionó y qué se verificó:** la primera configuración de fine-tuning (3 épocas congeladas, LR de backbone bajo) dio un resultado peor que el baseline entrenado desde cero. Se identificó la causa probable (backbone demasiado restringido para adaptarse a la señal supervisada), se ajustó el cronograma de descongelamiento, y se confirmó la mejora repitiendo el experimento con 3 semillas distintas antes de aceptar la configuración final — precisamente para no quedarse con un resultado que solo funcionara por azar en una corrida. El intento fallido se documentó en el reporte junto con el resultado final, en vez de omitirlo.

Este trabajo se dejó explícitamente marcado como borrador pendiente de revisión en `docs/CC3092_Proyecto2_1a1_y_division_trabajo.md`: las decisiones de diseño (arquitectura de la cabeza, estrategia de fine-tuning, función de pérdida) quedan sujetas a que el resto del equipo las revise y decida si las conserva, ajusta o rehace con su propio criterio.

