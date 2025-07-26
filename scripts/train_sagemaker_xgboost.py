er}"""
Train an XGBoost classifier on the credit risk dataset using SageMaker.
Uses all original features (except 'description') and the generated 'target' as label.
"""
import os
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from sagemaker import get_execution_role, Session
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

# Paths
DATA_PATH = 'data/credir_risk_reto_modified_prepared.csv'
S3_BUCKET = os.getenv('SAGEMAKER_S3_BUCKET')  # Set this in your .env
S3_PREFIX = 'credit-risk-xgboost'

# Load data
print('Loading data...')
df = pd.read_csv(DATA_PATH)

# Split features/target
X = df.drop(columns=['target'])
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save to CSV for SageMaker (no header, target first column)
train_df = pd.concat([y_train, X_train], axis=1)
test_df = pd.concat([y_test, X_test], axis=1)
train_file = 'data/xgb_train.csv'
test_file = 'data/xgb_test.csv'
train_df.to_csv(train_file, header=False, index=False)
test_df.to_csv(test_file, header=False, index=False)

# Upload to S3
session = boto3.Session()
sagemaker_session = Session(boto_session=session)
role = get_execution_role()
print('Uploading data to S3...')
train_s3 = sagemaker_session.upload_data(train_file, bucket=S3_BUCKET, key_prefix=S3_PREFIX)
test_s3 = sagemaker_session.upload_data(test_file, bucket=S3_BUCKET, key_prefix=S3_PREFIX)

# XGBoost estimator
xgb = XGBoost(
    entry_point=None,
    framework_version="1.7-1",
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f's3://{S3_BUCKET}/{S3_PREFIX}/output',
    role=role,
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "max_depth": 5,
        "eta": 0.2,
        "objective": "binary:logistic",
        "num_round": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
)

# Train
print('Starting XGBoost training on SageMaker...')
xgb.fit({
    "train": TrainingInput(train_s3, content_type="csv"),
    "validation": TrainingInput(test_s3, content_type="csv")
})
print('Training complete.')
