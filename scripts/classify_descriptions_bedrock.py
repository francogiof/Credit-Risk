"""
Clasifica las descripciones generadas usando AWS Bedrock y asigna etiquetas de riesgo ('target').
"""

import os
from dotenv import load_dotenv
import boto3
import pandas as pd
import json
import logging
import re
import time

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
        "\n\nHuman: Classify the following credit risk description as either 'good risk' or 'bad risk'. Respond with only one of these two options, no explanation, no judgment, and no extra words. Both answers are valid. Do not be pessimistic or biased toward 'bad risk'.\n"
        "Your decision must follow these known thresholds, derived from expert analysis of previous credit data:\n"
        "Good Credit Risk Indicators:\n"
        "- Job: 2 or 3 (skilled or highly skilled)\n"
        "- Saving Accounts: moderate, quite rich, rich\n"
        "- Checking Account: moderate, rich\n"
        "- Credit Amount: less than or equal to 3000\n"
        "- Duration: less than or equal to 18 months\n"
        "- Housing: own\n"
        "- Age: between 26 and 55 years\n"
        f"Description: {description}\n"
        "\nAssistant:"
    )
    print(f"Prompt for row {row.name}: {prompt}")
    retries = 3
    for attempt in range(retries):
        try:
            response = bedrock.invoke_model(
                modelId='anthropic.claude-v2',
                contentType='application/json',
                accept='application/json',
                body=json.dumps({
                    "prompt": prompt,
                    "max_tokens_to_sample": 10
                })
            )
            result = json.loads(response['body'].read())
            output = result.get('completion', '').strip().lower()
            print(f"LLM raw output for row {row.name}: {result.get('completion', '')}")
            # Remove punctuation and extra whitespace
            output_clean = re.sub(r'[^a-z ]', '', output)
            output_clean = output_clean.strip()
            # Strictly enforce only 'good risk' or 'bad risk'
            time.sleep(1)  # Wait 1 second after each request
            if output_clean == 'bad risk':
                return 'bad risk'
            elif output_clean == 'good risk':
                return 'good risk'
            else:
                return ''  # Leave blank for post-processing
        except Exception as e:
            if 'ThrottlingException' in str(e) and attempt < retries - 1:
                print(f"ThrottlingException for row {row.name}, retrying in 5 seconds...")
                time.sleep(5)
                continue
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
    targets = []
    for idx, row in df.iterrows():
        target = classify_risk_bedrock(row['description'], row)
        targets.append(target)
        df.at[idx, 'target'] = target
        df.to_csv('data/credir_risk_reto_modified.csv', index=False)  # Save after each row for robustness
    print('Risk classification completed and saved to credir_risk_reto_modified.csv')
    # Post-process to ensure only valid outputs
    postprocess_targets('data/credir_risk_reto_modified.csv')

if __name__ == "__main__":
    classify_descriptions()
