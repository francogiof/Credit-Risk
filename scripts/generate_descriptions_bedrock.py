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
import json
import logging

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

def get_description_bedrock(row):
    # Prepare field values, handling NaNs for 'Saving accounts' and 'Checking account'
    saving_acc = row['Saving accounts'] if pd.notnull(row['Saving accounts']) else None
    checking_acc = row['Checking account'] if pd.notnull(row['Checking account']) else None
    # Build context for prompt
    context = (
        "You are an expert in credit risk data documentation. "
        "Given the following person data, generate a strictly factual description as a single paragraph (no line breaks, no bullet points, no recommendations, no risk assessment, no subjective language). "
        "Include every field and value provided, and do not omit any information. "
        "Do not add any judgment, rating, or suggestion. "
        "If a field is NaN, do not mention it. "
        "Write the description strictly in English. "
        "Here are the field meanings for reference:\n"
        "Age: Edad de la persona\n"
        "Sex: Sexo de la persona\n"
        "Job: 0=unskilled and non-resident, 1=unskilled and resident, 2=skilled, 3=highly skilled\n"
        "Housing: Tipo de alojamiento\n"
        "Saving accounts: Tipo de cuenta de ahorro (may be NaN)\n"
        "Checking account: Tipo de cuenta corriente (may be NaN)\n"
        "Credit amount: Monto de crédito\n"
        "Duration: Tiempo de préstamo (meses)\n"
        "Purpose: Motivo del préstamo\n"
    )
    # Build person data string, omitting NaN fields
    person_data = {k: v for k, v in row.to_dict().items() if not (k in ['Saving accounts', 'Checking account'] and pd.isnull(v))}
    prompt = (
        f"\n\nHuman: {context} Person data: {person_data}\n\nAssistant:"
    )
    print(f"Prompt for row {row.name}: {prompt}")
    try:
        response = bedrock.invoke_model(
            modelId='anthropic.claude-v2:1',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 120
            })
        )
        result = json.loads(response['body'].read())
        description = result.get('completion', '').strip()
        # Remove line breaks and ensure only one paragraph
        description = ' '.join(description.splitlines()).replace('  ', ' ')
        print(f"Generated description for row {row.name}: {description}")
        return description
    except Exception as e:
        logging.error(f"Bedrock API error for row {row.name}: {e}")
        print(f"Error for row {row.name}: {e}")
        return ''

def generate_descriptions():
    df = pd.read_csv('data/credir_risk_reto.csv')
    df['description'] = df.apply(get_description_bedrock, axis=1)
    df.to_csv('data/credir_risk_reto_modified.csv', index=False)
    print('Descriptions generated and saved to credir_risk_reto_modified.csv')

if __name__ == "__main__":
    generate_descriptions()
