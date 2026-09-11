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
