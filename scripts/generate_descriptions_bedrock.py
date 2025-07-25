"""
Genera descripciones relacionadas con el riesgo crediticio usando AWS Bedrock (GPT, Llama).
"""
# TODO: Configurar credenciales y conexión con AWS Bedrock
# TODO: Leer el dataset original credir_risk_reto.csv
# TODO: Para cada fila, generar una descripción usando un modelo generativo (GPT, Llama) de Bedrock
# TODO: Añadir la columna 'description' al DataFrame
# TODO: Guardar el DataFrame modificado en credir_risk_reto_modified.csv
# TODO: Manejar errores y logs

import os
from dotenv import load_dotenv
import boto3
import pandas as pd

# Cargar variables de entorno
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# Configurar cliente de Bedrock
bedrock = boto3.client(
    'bedrock-runtime',
    region_name=AWS_DEFAULT_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def generate_descriptions():
    df = pd.read_csv('data/credir_risk_reto.csv')
    # ...aquí iría la llamada a Bedrock para cada fila...
    df['description'] = 'Descripción generada (mock)' # Placeholder
    df.to_csv('data/credir_risk_reto_modified.csv', index=False)

if __name__ == "__main__":
    generate_descriptions()
