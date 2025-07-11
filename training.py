# stock_ml_fixed.py
# Fixed Stock ML Project with correct SageMaker setup

import boto3
import pandas as pd
import numpy as np
import requests
import os
import time
import tarfile

# Configuration
BUCKET_NAME = "stock-ml-university-project-1751964028"  # Use existing bucket
AWS_REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::869508798872:role/SageMakerExecutionRole-ForexML"


def create_fixed_training_script():
    """Create a working training script for SageMaker"""
    print("📝 Creating fixed training script...")

    training_code = '''#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

def train_stock_model(args):
    """Train stock prediction model"""
    print("🚀 Starting Apple stock prediction training...")
    print(f"Model directory: {args.model_dir}")
    print(f"Training data directory: {args.train}")

    # List files in training directory
    train_files = os.listdir(args.train)
    print(f"Available files: {train_files}")

    # Find CSV file
    csv_file = None
    for file in train_files:
        if file.endswith('.csv'):
            csv_file = file
            break

    if csv_file is None:
        raise ValueError("No CSV file found in training directory")

    # Load data
    train_file_path = os.path.join(args.train, csv_file)
    print(f"Loading data from: {train_file_path}")

    df = pd.read_csv(train_file_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Prepare features and target
    feature_columns = [
        'open_price', 'high_price', 'low_price', 'volume',
        'price_change', 'price_range', 'prev_close',
        'moving_avg_5', 'moving_avg_10'
    ]

    # Check if all feature columns exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        # Use available numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target_price' in numeric_cols:
            numeric_cols.remove('target_price')
        feature_columns = numeric_cols[:min(9, len(numeric_cols))]  # Use up to 9 features
        print(f"Using available features: {feature_columns}")

    X = df[feature_columns]
    y = df['target_price']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")

    # Remove any rows with NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    print(f"After removing NaN - Features: {X.shape}, Target: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train models
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=50,  # Reduced for faster training
        max_depth=8,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    print("Training Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Evaluate models
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred = lr_model.predict(X_test)

    # Calculate metrics
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    rf_test_r2 = r2_score(y_test, rf_test_pred)

    lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
    lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
    lr_test_r2 = r2_score(y_test, lr_test_pred)

    print(f"\\nRandom Forest Results:")
    print(f"  Training RMSE: ${rf_train_rmse:.2f}")
    print(f"  Test RMSE: ${rf_test_rmse:.2f}")
    print(f"  Test R²: {rf_test_r2:.3f}")

    print(f"\\nLinear Regression Results:")
    print(f"  Training RMSE: ${lr_train_rmse:.2f}")
    print(f"  Test RMSE: ${lr_test_rmse:.2f}")
    print(f"  Test R²: {lr_test_r2:.3f}")

    # Choose best model
    if rf_test_rmse < lr_test_rmse:
        best_model = rf_model
        best_model_name = "RandomForest"
        best_rmse = rf_test_rmse
        best_r2 = rf_test_r2
    else:
        best_model = lr_model
        best_model_name = "LinearRegression"
        best_rmse = lr_test_rmse
        best_r2 = lr_test_r2

    print(f"\\nBest model: {best_model_name}")

    # Save best model
    model_path = os.path.join(args.model_dir, "model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Model saved: {model_path}")

    # Save feature names
    feature_path = os.path.join(args.model_dir, "feature_names.pkl")
    joblib.dump(list(X.columns), feature_path)

    # Save metrics
    metrics = {
        "model_type": best_model_name,
        "test_rmse": float(best_rmse),
        "test_r2": float(best_r2),
        "features_used": list(X.columns),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

    if best_model_name == "RandomForest":
        feature_importance = dict(zip(X.columns, best_model.feature_importances_))
        metrics["feature_importance"] = {k: float(v) for k, v in feature_importance.items()}

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Apple stock prediction training completed!")
    print(f"Final model: {best_model_name} with RMSE: ${best_rmse:.2f}, R²: {best_r2:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args = parser.parse_args()
    print(f"Arguments: model_dir={args.model_dir}, train={args.train}")

    try:
        train_stock_model(args)
    except Exception as e:
        print(f"Training error: {e}")
        raise
'''

    with open("train_stock_fixed.py", "w", encoding='utf-8') as f:
        f.write(training_code)

    print("✅ Fixed training script created")


def package_and_upload_fixed_code():
    """Package and upload fixed training code"""
    print("📦 Packaging fixed training code...")

    # Create source directory
    os.makedirs("stock_training_fixed", exist_ok=True)

    # Copy training script
    import shutil
    shutil.copy("train_stock_fixed.py", "stock_training_fixed/train.py")

    # Create requirements
    requirements = """scikit-learn==1.2.2
pandas==1.5.3
numpy==1.21.0
joblib==1.2.0
"""

    with open("stock_training_fixed/requirements.txt", "w") as f:
        f.write(requirements)

    # Package
    with tarfile.open("stock_training_fixed.tar.gz", "w:gz") as tar:
        tar.add("stock_training_fixed", arcname=".")

    # Upload to S3
    s3 = boto3.client('s3', region_name=AWS_REGION)
    timestamp = int(time.time())
    s3_key = f"training-code/stock_training_fixed_{timestamp}.tar.gz"

    s3.upload_file("stock_training_fixed.tar.gz", BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"

    print(f"✅ Fixed training code uploaded: {s3_uri}")
    return s3_uri


def create_fixed_sagemaker_job(training_code_uri):
    """Create SageMaker job with correct container"""
    print("🚀 Creating fixed SageMaker training job...")

    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
    timestamp = int(time.time())
    job_name = f"stock-prediction-fixed-{timestamp}"

    # Use the correct scikit-learn container for us-east-1
    image_uri = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"

    config = {
        'TrainingJobName': job_name,
        'AlgorithmSpecification': {
            'TrainingImage': image_uri,
            'TrainingInputMode': 'File'
        },
        'RoleArn': ROLE_ARN,
        'InputDataConfig': [{
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f"s3://{BUCKET_NAME}/training-data/",
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'text/csv'
        }],
        'OutputDataConfig': {
            'S3OutputPath': f"s3://{BUCKET_NAME}/model-output-fixed/"
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 20
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 1800
        },
        'HyperParameters': {
            'sagemaker_program': 'train.py',
            'sagemaker_submit_directory': training_code_uri
        }
    }

    try:
        response = sagemaker.create_training_job(**config)
        print(f"✅ Fixed training job created: {job_name}")
        return job_name
    except Exception as e:
        print(f"❌ Training job creation failed: {e}")
        return None


def monitor_training_job(job_name):
    """Monitor training progress"""
    print(f"👀 Monitoring training: {job_name}")

    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)

    for i in range(40):  # 20 minutes max
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']

            print(f"Status: {status}")

            if status == 'Completed':
                print("🎉 Training completed successfully!")
                return response['ModelArtifacts']['S3ModelArtifacts']
            elif status == 'Failed':
                print("❌ Training failed!")
                if 'FailureReason' in response:
                    print(f"Reason: {response['FailureReason']}")
                return None
            elif status == 'Stopped':
                print("⏹️ Training stopped")
                return None

            time.sleep(30)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

    print("⏰ Training timeout")
    return None


def main():
    """Main fixed workflow"""
    print("🔧 FIXING STOCK PREDICTION ML PROJECT")
    print("=" * 60)

    # Step 1: Create fixed training script
    create_fixed_training_script()

    # Step 2: Package and upload fixed code
    training_code_uri = package_and_upload_fixed_code()

    # Step 3: Create fixed training job
    job_name = create_fixed_sagemaker_job(training_code_uri)
    if job_name is None:
        print("❌ Failed to create training job")
        return

    # Step 4: Monitor training
    model_artifacts = monitor_training_job(job_name)

    if model_artifacts:
        print("\n" + "=" * 60)
        print("🎉 FIXED STOCK PREDICTION PROJECT SUCCESS!")
        print("=" * 60)
        print(f"Training Job: {job_name}")
        print(f"Model Artifacts: {model_artifacts}")
        print("Algorithm: Random Forest + Linear Regression")
        print("Data: Apple Stock (990 samples)")
        print("Status: WORKING ML MODEL! 📈")
        print("Perfect for university presentation!")
        print("=" * 60)
    else:
        print("❌ Training still failed")


if __name__ == "__main__":
    main()