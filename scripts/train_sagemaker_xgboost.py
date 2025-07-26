"""
Train an XGBoost classifier on the credit risk dataset using SageMaker.
- Data preparation and job submission run locally.
- Model training runs remotely on AWS SageMaker.
"""

import os
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from sagemaker import get_execution_role, Session
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = 'data/credir_risk_reto_modified_prepared.csv'
S3_BUCKET = os.getenv('SAGEMAKER_S3_BUCKET')
S3_PREFIX = 'credit-risk-xgboost'
ROLE = os.getenv('SAGEMAKER_ROLE_ARN')

# 1. Load data (local)
print('Loading data...')
df = pd.read_csv(DATA_PATH)

# 2. Encode categorical/ordinal columns (local)
categorical_cols = ['Sex', 'Housing', 'Purpose']
ordinal_cols = {
    'Job': None,  # already ordinal
    'Saving accounts': None,  # already ordinal
    'Checking account': None  # already ordinal
}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
# (Ordinal columns assumed already encoded as integers)

# 3. Split features/target (local)
X = df.drop(columns=['target'])
y = df['target']

# 4. Train/validation/test split (local)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# 5. Save to CSV for SageMaker (no header, target first column) (local)
def save_for_sagemaker(X, y, path):
    pd.concat([y, X], axis=1).to_csv(path, header=False, index=False)
train_file = 'data/xgb_train.csv'
val_file = 'data/xgb_val.csv'
test_file = 'data/xgb_test.csv'
save_for_sagemaker(X_train, y_train, train_file)
save_for_sagemaker(X_val, y_val, val_file)
save_for_sagemaker(X_test, y_test, test_file)

# 6. Upload to S3 (local)
session = boto3.Session()
sagemaker_session = Session(boto_session=session)
role = ROLE or get_execution_role()
print('Uploading data to S3...')
train_s3 = sagemaker_session.upload_data(train_file, bucket=S3_BUCKET, key_prefix=S3_PREFIX)
val_s3 = sagemaker_session.upload_data(val_file, bucket=S3_BUCKET, key_prefix=S3_PREFIX)
test_s3 = sagemaker_session.upload_data(test_file, bucket=S3_BUCKET, key_prefix=S3_PREFIX)

# 7. Configure XGBoost estimator (local, but training runs in SageMaker)
xgb = XGBoost(
    entry_point=None,  # Using built-in XGBoost
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

# 8. Launch training job (runs remotely in SageMaker)
print('Starting XGBoost training on SageMaker...')
try:
    xgb.fit({
        "train": TrainingInput(train_s3, content_type="csv"),
        "validation": TrainingInput(val_s3, content_type="csv")
    })
    print('Training complete.')
except Exception as e:
    print(f'Error during training: {e}')

# 9. Save test set for later evaluation (local)
print(f'Test set saved to {test_file}. You can use this for post-training evaluation.')

# 10. (Optional) Download model artifact or evaluate after training (not shown here)
# See SageMaker docs for model artifact download and inference.
