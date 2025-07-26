"""
Train an XGBoost classifier on the credit risk dataset using SageMaker.
Uses all original features (except 'description') and the generated 'target' as label.
"""
import os
import pandas as pd
import numpy as np
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = "../data/credir_risk_reto_modified.csv"
TRAIN_PATH = "../data/train.csv"
TEST_PATH = "../data/test.csv"

# 1. Load and preprocess data

df = pd.read_csv(DATA_PATH)
# Drop rows with missing target or features
feature_cols = [
    'Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account',
    'Credit amount', 'Duration', 'Purpose'
]
df = df.dropna(subset=feature_cols + ['target'])

# Encode categorical variables
for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    df[col] = df[col].astype(str)
    df[col] = df[col].fillna('missing')
    df[col] = pd.Categorical(df[col]).codes

# Encode target
label_map = {'good risk': 1, 'bad risk': 0}
df['target'] = df['target'].map(label_map)

# 2. Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
train_df.to_csv(TRAIN_PATH, index=False, header=False)
test_df.to_csv(TEST_PATH, index=False, header=False)

# 3. Upload to S3
session = sagemaker.Session()
role = get_execution_role()
bucket = session.default_bucket()
train_s3 = session.upload_data(TRAIN_PATH, bucket=bucket, key_prefix='credit-risk-xgb')
test_s3 = session.upload_data(TEST_PATH, bucket=bucket, key_prefix='credit-risk-xgb')

# 4. Set up XGBoost estimator
xgb = XGBoost(
    entry_point=None,
    framework_version="1.5-1",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    output_path=f"s3://{bucket}/credit-risk-xgb/output",
    role=role,
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100,
        "max_depth": 5,
        "eta": 0.2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "auc"
    },
    sagemaker_session=session
)

# 5. Train
xgb.fit({
    "train": TrainingInput(train_s3, content_type="csv"),
    "validation": TrainingInput(test_s3, content_type="csv")
})

print("Training complete. Model artifacts saved to:", xgb.model_data)
