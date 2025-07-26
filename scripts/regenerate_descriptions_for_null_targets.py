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
    # Anthropic Claude v2.1 requires prompt to start with '\n\nHuman:' and end with '\n\nAssistant:'
    prompt = (
        "\n\nHuman: You are a data analyst. Given the following applicant data, generate a strictly factual, single-paragraph, English-only description of the applicant. "
        "Do not include any opinions, apologies, or extra commentary. Only use the data provided.\n\n"
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

def generate_description(row, max_retries=3, sleep_time=2):
    prompt = build_prompt(row)
    body = {"prompt": prompt, "max_tokens_to_sample": 200, "temperature": 0.2}
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
            return description
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(sleep_time)
    return None

if __name__ == "__main__":
    input_path = 'data/credir_risk_reto_modified.csv'
    output_path = 'data/credir_risk_reto_modified.csv'
    indices_to_fix = [244, 246, 281, 308, 575, 723, 939]

    df = pd.read_csv(input_path)

    for idx in indices_to_fix:
        if idx >= len(df):
            print(f"Index {idx} out of range.")
            continue
        print(f"Regenerating description for row {idx}...")
        row = df.iloc[idx]
        description = generate_description(row)
        if description:
            df.at[idx, 'description'] = description
            print(f"Row {idx} new description: {description[:80]}...")
            df.to_csv(output_path, index=False)
        else:
            print(f"[ERROR] Could not generate description for row {idx}.")
        time.sleep(1)
    print("Descriptions regenerated for specified rows.")
