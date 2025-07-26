"""
Clasifica las descripciones generadas usando AWS Bedrock y asigna etiquetas de riesgo ('target').
"""

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

def classify_risk_bedrock(description, row):
    prompt = (
        "\n\nHuman: You are an expert in credit risk analysis. "
        "Given the following credit risk description, classify the risk as either 'bad risk' or 'good risk'. "
        "Only use the information provided. Avoid hallucinations.\n"
        f"Description: {description}\n"
        "\nAssistant:"
    )
    print(f"Prompt for row {row.name}: {prompt}")
    try:
        response = bedrock.invoke_model(
            modelId='anthropic.claude-v2:1',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 20
            })
        )
        result = json.loads(response['body'].read())
        output = result.get('completion', '').strip().lower()
        print(f"Model output for row {row.name}: {output}")
        # Strict mapping: only accept exact or very close variants
        if output.startswith('bad risk'):
            return 'bad risk'
        elif output.startswith('good risk'):
            return 'good risk'
        # Sometimes model may return just 'bad' or 'good'
        elif output.strip() == 'bad':
            return 'bad risk'
        elif output.strip() == 'good':
            return 'good risk'
        else:
            return ''  # Leave blank for post-processing
    except Exception as e:
        logging.error(f"Bedrock API error for row {row.name}: {e}")
        print(f"Error for row {row.name}: {e}")
        return ''

def postprocess_targets(csv_path):
    df = pd.read_csv(csv_path)
    valid = ['good risk', 'bad risk']
    before = len(df)
    # Clean up any unexpected outputs
    df['target'] = df['target'].apply(lambda x: x if isinstance(x, str) and x.strip().lower() in valid else '')
    cleaned = df['target'].value_counts().to_dict()
    after = df['target'].isin(valid).sum()
    print(f"Post-processing: {before-after} rows cleaned. Counts: {cleaned}")
    df.to_csv(csv_path, index=False)

def classify_descriptions():
    df = pd.read_csv('data/credir_risk_reto_modified.csv')
    df['target'] = df.apply(lambda row: classify_risk_bedrock(row['description'], row), axis=1)
    df.to_csv('data/credir_risk_reto_modified.csv', index=False)
    print('Risk classification completed and saved to credir_risk_reto_modified.csv')
    # Post-process to ensure only valid outputs
    postprocess_targets('data/credir_risk_reto_modified.csv')

if __name__ == "__main__":
    classify_descriptions()
