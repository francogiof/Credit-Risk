import os
import time
import pandas as pd
import boto3
import json
from dotenv import load_dotenv

# Load environment variables for AWS credentials
load_dotenv()

# AWS Bedrock client setup
bedrock = boto3.client(
    'bedrock-runtime',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('AWS_SESSION_TOKEN')
)

MODEL_ID = 'anthropic.claude-v2:1'

def build_prompt(row):
    # Stronger prompt for more detail, no generic phrases
    prompt = (
        "\n\nHuman: You are a data analyst. Given the following applicant data, generate a strictly factual, detailed, single-paragraph, English-only description of the applicant. Do not include any opinions, apologies, or generic phrases. Use all available data fields. If a field is missing, simply omit it. Do not start with 'Here is' or similar.\n\n"
        f"Data: {row.to_dict()}\n\nAssistant:"
    )
    return prompt

def extract_claude_completion(raw_response):
    # Handles Anthropic Claude completion JSON string
    try:
        if raw_response.strip().startswith('{'):
            import json
            data = json.loads(raw_response)
            return data.get('completion', '').strip()
        # Fallback: try to extract after 'completion":'
        if 'completion":"' in raw_response:
            return raw_response.split('completion":"',1)[1].split('"',1)[0].strip()
    except Exception as e:
        print(f"[WARN] Could not extract completion: {e}")
    return raw_response.strip()

def generate_description(row, max_retries=7, sleep_time=8):
    prompt = build_prompt(row)
    body = {"prompt": prompt, "max_tokens_to_sample": 500, "temperature": 0.1}
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(body).encode('utf-8'),
                accept='application/json',
                contentType='application/json'
            )
            result = response['body'].read().decode('utf-8').strip()
            description = extract_claude_completion(result)
            # Only keep the first paragraph
            description = description.split('\n')[0].strip()
            # Remove generic phrases if present
            for bad_start in ["Here is", "Unfortunately", "I do not", "The data provided", "This data", "Strictly factual description:"]:
                if description.lower().startswith(bad_start.lower()):
                    description = description[len(bad_start):].lstrip(':,. ')
            return description
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(sleep_time)
    return None

def build_classification_prompt(description):
    # Strong, strict prompt for classification
    prompt = (
        "\n\nHuman: You are a credit risk expert. Given the following applicant description, classify the credit risk as strictly either 'good risk' or 'bad risk'. Respond with only one of these two options, nothing else. Use all available information in the description.\n\n"
        f"Description: {description}\n\nAssistant:"
    )
    return prompt

def classify_target(description, max_retries=7, sleep_time=8):
    prompt = build_classification_prompt(description)
    body = {"prompt": prompt, "max_tokens_to_sample": 10, "temperature": 0.0}
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(body).encode('utf-8'),
                accept='application/json',
                contentType='application/json'
            )
            result = response['body'].read().decode('utf-8').strip()
            label = extract_claude_completion(result).lower()
            if 'good risk' in label:
                return 'good risk'
            elif 'bad risk' in label:
                return 'bad risk'
            else:
                print(f"[WARN] Unexpected output: {label}")
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(sleep_time)
    return None

if __name__ == "__main__":
    input_path = 'data/credir_risk_reto_modified.csv'
    output_path = 'data/credir_risk_reto_modified.csv'
    
    df = pd.read_csv(input_path)
    # Find all indices with null target
    indices_to_fix = df[df['target'].isnull()].index.tolist()
    print(f"Null target indices: {indices_to_fix}")

    # Regenerate description and classify target for all nulls
    for idx in indices_to_fix:
        if idx >= len(df):
            print(f"Index {idx} out of range.")
            continue
        # Regenerate description
        print(f"Regenerating description for row {idx}...")
        row = df.iloc[idx]
        description = generate_description(row)
        if description:
            df.at[idx, 'description'] = description
            print(f"Row {idx} new description: {description[:120]}...")
            df.to_csv(output_path, index=False)
        else:
            print(f"[ERROR] Could not generate description for row {idx}.")
        time.sleep(2)
        # Classify target
        print(f"Classifying target for row {idx}...")
        label = classify_target(df.at[idx, 'description'])
        if label:
            df.at[idx, 'target'] = label
            print(f"Row {idx} classified as: {label}")
            df.to_csv(output_path, index=False)
        else:
            print(f"[ERROR] Could not classify row {idx}.")
        time.sleep(2)
    print("Descriptions and targets regenerated/classified for all nulls.")
