# 🚀 Challenge AI: Detección de Fraudes en Transacciones Bancarias

## 🎯 Objetivo

Este proyecto aborda la detección de fraudes en transacciones bancarias utilizando AWS Bedrock y Amazon SageMaker. El flujo completo incluye generación de descripciones, clasificación de riesgo, entrenamiento y despliegue de modelos escalables.

### Pasos principales:

1. **Generación de descripciones**  
   Utiliza AWS Bedrock (GPT, Llama) para crear descripciones (`description`) relacionadas con el riesgo crediticio a partir del dataset `credir_risk_reto.csv`.

2. **Clasificación inicial**  
   Clasifica las descripciones generadas con Bedrock, asignando etiquetas de riesgo (`target`): “bad risk” o “good risk”.

3. **Entrenamiento supervisado**  
   Entrena un modelo de clasificación en Amazon SageMaker usando las etiquetas generadas, optimizando hiperparámetros para el mejor rendimiento.

4. **Despliegue y monitoreo**  
   Despliega el modelo en SageMaker para procesar nuevas transacciones y monitorear desempeño. El sistema es eficiente, escalable y adaptable a nueva data.

---

## 📦 Estructura del Proyecto

```
credit-risk-classifier/
├── scripts/
│   ├── EDA.ipynb                         # Análisis exploratorio de datos (EDA)
│   ├── generate_descriptions_bedrock.py   # Generación de descripciones con Bedrock
│   ├── classify_descriptions_bedrock.py   # Clasificación inicial con Bedrock
│   ├── train_sagemaker.py                 # Entrenamiento en SageMaker
│   ├── deploy_sagemaker.py                # Despliegue y pruebas de inferencia
│   └── query_endpoint.py                  # Consulta al modelo desplegado
├── data/
│   └── credir_risk_reto.csv               # Dataset original
│   └── credir_risk_reto_modified.csv      # Dataset con columnas description y target
├── logs/                                  # Logs de ejecución y entrenamiento
├── results/                               # Métricas y reportes
├── README.md                              # Instrucciones y documentación
└── requirements.txt                       # Dependencias Python
```

---

## 📝 Diccionario de Datos

- **Age**: Edad de la persona
- **Sex**: Sexo
- **Job**: 0 (no calificado/no residente), 1 (no calificado/residente), 2 (calificado), 3 (altamente calificado)
- **Housing**: Tipo de alojamiento
- **Saving accounts**: Tipo de cuenta de ahorro
- **Checking account**: Tipo de cuenta corriente
- **Credit amount**: Monto de crédito
- **Duration (meses)**: Tiempo de préstamo
- **Purpose**: Motivo del préstamo
- **description**: Descripción generada por Bedrock (nueva columna)
- **target**: Etiqueta de riesgo (“bad risk” o “good risk”, nueva columna)

---

## ⚡️ Getting Started

1. Realiza el análisis exploratorio:
   ```bash
   jupyter notebook scripts/EDA.ipynb
   ```
2. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Genera descripciones con Bedrock:
   ```bash
   python scripts/generate_descriptions_bedrock.py
   ```

4. Clasifica descripciones:
   ```bash
   python scripts/classify_descriptions_bedrock.py
   ```

5. Entrena el modelo en SageMaker:
   ```bash
   python scripts/train_sagemaker.py
   ```

6. Despliega y consulta el endpoint:
   ```bash
   python scripts/deploy_sagemaker.py
   python scripts/query_endpoint.py
   ```

---

## 📊 Entregables

- Scripts funcionales para cada etapa del flujo.
- Informe corto:
  - Descripción del workflow
  - Decisiones técnicas
  - Métricas de desempeño
  - Screenshots/logs de Bedrock y SageMaker
- Enlace o evidencia del endpoint desplegado.
- Script/instrucciones para consultas.
- Dataset modificado con nuevas columnas.
- Video presentando hallazgos y prototipo.

---

## 🛠️ Tecnologías Clave

- AWS Bedrock (GPT, Llama)
- Amazon SageMaker
- Python (boto3, pandas, scikit-learn)
- Jupyter Notebook

---

## 📄 License

MIT License

---