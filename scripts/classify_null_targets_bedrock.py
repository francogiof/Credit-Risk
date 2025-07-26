import os
import time
import pandas as pd
import boto3
import json
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# Load environment variables for AWS credentials
load_dotenv()

# AWS Bedrock client setup
def get_bedrock_client():
    return boto3.client(
        'bedrock-runtime',
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN')
    )

# Prompt for strict classification
def build_prompt(description):
    return (
        "You are a credit risk expert. Given the following applicant description, classify the credit risk as strictly either 'good risk' or 'bad risk'. "
        "Respond with only one of these two options, nothing else.\n\n"
        f"Description: {description}\n\nClassification:"
    )

# Call Bedrock LLM for classification
def classify_description(client, description, model_id):
    prompt = build_prompt(description)
    body = {"prompt": prompt, "max_tokens_to_sample": 10, "temperature": 0.0}
    for attempt in range(3):
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body).encode('utf-8'),  # Proper JSON serialization
                accept='application/json',
                contentType='application/json'
            )
            result = response['body'].read().decode('utf-8')
            # Extract only the first valid label
            if 'good risk' in result.lower():
                return 'good risk'
            elif 'bad risk' in result.lower():
                return 'bad risk'
            else:
                print(f"[WARN] Unexpected output: {result}")
        except (BotoCoreError, ClientError) as e:
            print(f"[ERROR] Bedrock call failed: {e}. Retrying...")
            time.sleep(2 ** attempt)
    return None

if __name__ == "__main__":
    # File paths
    input_csv = 'data/credir_risk_reto_modified.csv'
    output_csv = 'data/credir_risk_reto_modified.csv'  # Overwrite or change as needed

    # Indices of rows with null target
    null_indices = [244, 246, 281, 308, 575, 723, 939]

    # Bedrock model ID (update as needed)
    bedrock_model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-v2')

    # Load data
    df = pd.read_csv(input_csv)

    # Connect to Bedrock
    client = get_bedrock_client()

    # Process only rows with null target at specified indices
    for idx in null_indices:
        row = df.iloc[idx]
        if pd.isnull(row['target']):
            description = row['description']
            print(f"Classifying row {idx}...")
            label = classify_description(client, description, bedrock_model_id)
            if label:
                df.at[idx, 'target'] = label
                print(f"Row {idx} classified as: {label}")
            else:
                print(f"[ERROR] Could not classify row {idx}. Leaving as null.")
            time.sleep(1)  # Throttle requests

    # Save updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Updated file saved to {output_csv}")
