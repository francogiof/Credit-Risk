# 📝 Assignment Guide: Credit Risk Classifier Challenge

Este documento es una guía paso a paso para completar los TODOs en los scripts del proyecto, explicando el propósito y la lógica detrás de cada etapa.

---

## 0. Análisis Exploratorio de Datos (EDA)
**Script:** `scripts/EDA.ipynb`

### Objetivo
Explorar y entender el dataset antes de procesar y modelar.

### Pasos
- Abre el notebook EDA.ipynb en Jupyter.
- Carga el archivo `data/credir_risk_reto.csv`.
- Analiza estadísticas descriptivas, distribuciones y correlaciones.
- Identifica valores nulos, outliers y patrones relevantes.
- Documenta hallazgos y posibles transformaciones necesarias.

**¿Por qué?**
El EDA permite comprender la estructura y calidad de los datos, facilitando decisiones informadas para el preprocesamiento y modelado.

---

## 1. Generación de Descripciones con AWS Bedrock
**Script:** `scripts/generate_descriptions_bedrock.py`

### Objetivo
Generar una columna `description` en el dataset usando modelos generativos (GPT, Llama) de AWS Bedrock.

### Pasos
- Configura las credenciales y la conexión con AWS Bedrock (usando boto3).
- Lee el archivo `data/credir_risk_reto.csv`.
- Para cada fila, envía los datos relevantes al modelo generativo y recibe una descripción.
- Añade la columna `description` al DataFrame.
- Guarda el resultado en `data/credir_risk_reto_modified.csv`.
- Registra errores y logs para trazabilidad.

**¿Por qué?**
Las descripciones generadas enriquecen el dataset y permiten que el modelo capture patrones de riesgo crediticio en lenguaje natural.

---

## 2. Clasificación Inicial con AWS Bedrock
**Script:** `scripts/classify_descriptions_bedrock.py`

### Objetivo
Clasificar cada descripción generada como "bad risk" o "good risk" usando Bedrock.

### Pasos
- Configura la conexión con AWS Bedrock.
- Lee el archivo `data/credir_risk_reto_modified.csv`.
- Para cada descripción, solicita al modelo una clasificación de riesgo.
- Añade la columna `target` al DataFrame.
- Guarda el resultado actualizado.
- Registra errores y logs.

**¿Por qué?**
La clasificación automática permite crear etiquetas para entrenamiento supervisado, simulando un sistema experto.

---

## 3. Entrenamiento Supervisado en SageMaker
**Script:** `scripts/train_sagemaker.py`

### Objetivo
Entrenar un modelo de clasificación usando las descripciones y etiquetas generadas.

### Pasos
- Configura la conexión con SageMaker.
- Lee el dataset modificado.
- Preprocesa los datos (vectorización, split train/test).
- Define el modelo y los hiperparámetros.
- Entrena el modelo en SageMaker.
- Guarda métricas y logs en `results/` y `logs/`.
- Almacena el modelo entrenado en S3 o Model Registry.

**¿Por qué?**
El entrenamiento supervisado permite crear un modelo robusto y escalable para predecir riesgo crediticio en nuevas transacciones.

---

## 4. Despliegue y Pruebas de Inferencia
**Script:** `scripts/deploy_sagemaker.py`

### Objetivo
Desplegar el modelo entrenado como endpoint en SageMaker y validar su funcionamiento.

### Pasos
- Configura la conexión con SageMaker.
- Selecciona el modelo entrenado.
- Despliega el modelo como endpoint.
- Realiza pruebas de inferencia con datos de ejemplo.
- Guarda logs y resultados en `results/`.

**¿Por qué?**
El despliegue permite que el modelo esté disponible para consultas en tiempo real, facilitando la integración con sistemas bancarios.

---

## 5. Consulta al Endpoint Desplegado
**Script:** `scripts/query_endpoint.py`

### Objetivo
Realizar inferencias sobre nuevas transacciones usando el endpoint desplegado.

### Pasos
- Configura la conexión con el endpoint de SageMaker.
- Prepara los datos de entrada para la consulta.
- Realiza la consulta y obtiene la predicción.
- Muestra el resultado y guarda logs en `results/`.

**¿Por qué?**
Permite validar el modelo en producción y realizar predicciones sobre nuevos casos de riesgo crediticio.

---

## Recomendaciones Generales
- Documenta cada paso y decisión técnica en los scripts y en los logs.
- Maneja errores y excepciones para asegurar robustez.
- Usa variables de entorno para credenciales y configuraciones sensibles.
- Realiza pruebas unitarias en cada script.

---

Sigue esta guía para completar los TODOs y lograr un flujo funcional, escalable y trazable para la detección de fraudes en transacciones bancarias.
