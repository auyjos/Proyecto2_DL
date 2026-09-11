# Registro de prompts de IA — José Auyón

Este archivo conserva los prompts utilizados en la preparación de C1. El reporte final debe resumir el uso de IA del equipo completo en un máximo de 200 palabras.

## Planificación

**Prompt:** «necesito que me ayudes planificando como vamos a hacer mis cambios yo soy Jose Auyon».

**Finalidad:** convertir la asignación general en un contrato de datos, tareas, pruebas y fechas. Funcionó porque el documento ya indicaba el ownership de José y permitió contrastarlo con los archivos disponibles. José eligió IBM para las secuencias y el formato notebook más módulo.

## Comparación reproducible

**Prompt:** «Puedes también dejar documentado a nivel código como se hicieron estas comparaciones, eso nos ayuda a visualizar las diferencias».

**Finalidad:** pasar las mediciones exploratorias a scripts, pruebas, tablas y gráficas reproducibles. Funcionó porque las definiciones de remitente, percentil, muestreo y recorte ya se habían acordado. Las cifras fueron calculadas desde los CSV completos.

## Variantes IBM

**Prompts:** «Que pasa con los demás csvs que son más pessados como large trans y accounts? medium también» y «Añadimos esto como parte del análisis también?».

**Finalidad:** comparar HI/LI en Small, Medium y Large, y auditar la relación entre cuentas y entidades. La revisión mostró que el cruce requiere normalizar ceros iniciales del banco y que muchas entidades poseen varias cuentas. El equipo debe decidir cómo usar los resultados experimentales; el código no presupone que el archivo más grande sea mejor.

## Implementación

**Prompt:** «Listo, el plan fue aceptado por favor ayudame a implementarlo».

**Finalidad:** implementar el pipeline aprobado, sus pruebas, artefactos, notebook y documentación. La IA propuso agrupar splits por `Entity ID` a partir del hallazgo aceptado del análisis; José y el equipo conservan la decisión sobre arquitectura, pérdidas, umbral y modelos posteriores.

## Organización de la rama

**Prompts:** «haz una rama con mejor nombre siguiendo las mejores convenciones de software» y «no quita el codex».

**Finalidad:** separar el trabajo de C1 en una rama descriptiva y fácil de revisar. El resultado fue `feature/c1-sequence-data-pipeline`; José indicó explícitamente que no quería el prefijo sugerido por la herramienta.

## Descripción del proyecto

**Prompt:** «puedes mejorar un poco e readme para decir de que trata el proyecto».

**Finalidad:** ampliar el README con el problema que se busca resolver, el sistema de detección en dos etapas, el papel de la interpretabilidad, la elección de HI-Small y el alcance implementado de C1. La redacción se contrastó con el enunciado y con el código existente para no presentar C2A, C2B ni el MVP como terminados.

## Actualización del registro y la división de trabajo

**Prompt:** «me guardas los promts por favor y me actualizas la división de trabajo a partir de lo que hemos hecho».

**Finalidad:** conservar las solicitudes posteriores a la implementación y actualizar el estado del equipo usando evidencia verificable del repositorio. Las tareas de C1 se marcaron como completadas; las responsabilidades de Ruiz, Gerardo y el cierre grupal permanecen pendientes de integración en esta rama.
