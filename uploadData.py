# stock_ml_project.py
# Simple Stock Price Prediction ML Project for University

import boto3
import pandas as pd
import numpy as np
import requests
import os
import time
import tarfile
from datetime import datetime, timedelta

# Configuration
BUCKET_NAME = "stock-ml-university-project"
AWS_REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::869508798872:role/SageMakerExecutionRole-ForexML"


def fetch_stock_data():
    """Fetch Apple stock data using Twelve Data API"""
    print("📈 Fetching Apple (AAPL) stock data from Twelve Data...")

    import requests

    TWELVE_DATA_API_KEY = "a6cab3620d2147dbbf1bf4fede6463f1"
    TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"

    try:
        # Get Apple stock data
        params = {
            "symbol": "AAPL",
            "interval": "1day",
            "outputsize": "1000",  # Get lots of data
            "apikey": TWELVE_DATA_API_KEY,
            "format": "JSON"
        }

        print("Requesting data from Twelve Data API...")
        response = requests.get(TWELVE_DATA_BASE_URL, params=params)
        data = response.json()

        if "error" in data:
            raise Exception(f"API Error: {data['error']}")

        if "values" not in data:
            raise Exception(f"No values in response: {data}")

        print(f"✅ Received {len(data['values'])} data points from API")

        # Convert to DataFrame
        df = pd.DataFrame(data['values'])

        # Convert columns to proper types
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(int)

        # Sort by date (oldest first)
        df = df.sort_values('datetime').reset_index(drop=True)

        # Rename columns for ML
        df.columns = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']

        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Price range: ${df['close_price'].min():.2f} - ${df['close_price'].max():.2f}")

        # Create ML features
        df['price_change'] = df['close_price'] - df['open_price']
        df['price_range'] = df['high_price'] - df['low_price']
        df['prev_close'] = df['close_price'].shift(1)
        df['moving_avg_5'] = df['close_price'].rolling(window=5).mean()
        df['moving_avg_10'] = df['close_price'].rolling(window=10).mean()

        # Remove NaN values
        df = df.dropna()

        # Save to CSV
        df.to_csv('apple_stock_data.csv', index=False)
        print(f"✅ Saved {len(df)} rows to apple_stock_data.csv")

        return df

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


def prepare_ml_data(df):
    """Prepare data for machine learning"""
    print("🔧 Preparing data for ML training...")

    # Features for prediction
    feature_columns = [
        'open_price', 'high_price', 'low_price', 'volume',
        'price_change', 'price_range', 'prev_close',
        'moving_avg_5', 'moving_avg_10'
    ]

    # Target: next day's closing price
    df['target_price'] = df['close_price'].shift(-1)

    # Remove last row (no target) and any NaN
    df = df.dropna()

    # Create training dataset
    ml_data = df[feature_columns + ['target_price']].copy()

    print(f"Features: {feature_columns}")
    print(f"Target: next_day_close_price")
    print(f"Training samples: {len(ml_data)}")

    # Save ML-ready data
    ml_data.to_csv('stock_ml_training_data.csv', index=False)
    print("✅ ML training data saved")

    return ml_data


def create_s3_bucket():
    """Create S3 bucket for the project"""
    print("🪣 Creating S3 bucket...")

    s3 = boto3.client('s3', region_name=AWS_REGION)

    try:
        bucket_name = f"{BUCKET_NAME}-{int(time.time())}"

        s3.create_bucket(Bucket=bucket_name)
        print(f"✅ Created bucket: {bucket_name}")

        return bucket_name

    except Exception as e:
        print(f"❌ Error creating bucket: {e}")
        return None


def upload_data_to_s3(bucket_name):
    """Upload training data to S3"""
    print("📤 Uploading data to S3...")

    s3 = boto3.client('s3', region_name=AWS_REGION)

    try:
        # Upload training data
        s3.upload_file(
            'stock_ml_training_data.csv',
            bucket_name,
            'training-data/stock_data.csv'
        )

        s3_uri = f"s3://{bucket_name}/training-data/stock_data.csv"
        print(f"✅ Data uploaded: {s3_uri}")

        return s3_uri

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None


def create_training_script():
    """Create simple ML training script"""
    print("📝 Creating training script...")

    training_code = '''
#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

def train_model(args):
    """Train Random Forest model for stock prediction"""
    print("🚀 Starting stock prediction training...")

    # Load data
    train_file = os.path.join(args.train, "stock_data.csv")
    df = pd.read_csv(train_file)

    print(f"Loaded {len(df)} training samples")
    print(f"Features: {list(df.columns[:-1])}")

    # Prepare features and target
    X = df.iloc[:, :-1]  # All columns except last (target)
    y = df.iloc[:, -1]   # Last column (target_price)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate model
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")

    # Feature importance
    feature_names = X.columns
    importance = model.feature_importances_

    print("\\nFeature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.3f}")

    # Save model
    model_path = os.path.join(args.model_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    # Save feature names
    feature_path = os.path.join(args.model_dir, "feature_names.pkl")
    joblib.dump(list(feature_names), feature_path)

    # Save metrics
    metrics = {
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "feature_importance": {name: float(imp) for name, imp in zip(feature_names, importance)}
    }

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args = parser.parse_args()
    train_model(args)
'''

    with open("train_stock_model.py", "w", encoding='utf-8') as f:
        f.write(training_code)

    print("✅ Training script created: train_stock_model.py")


def package_training_code():
    """Package training code for SageMaker"""
    print("📦 Packaging training code...")

    # Create source directory
    os.makedirs("stock_training_code", exist_ok=True)

    # Copy training script
    import shutil
    shutil.copy("train_stock_model.py", "stock_training_code/train.py")

    # Create requirements
    requirements = """scikit-learn==1.3.0
pandas==1.5.3
numpy==1.24.3
joblib==1.3.2
"""

    with open("stock_training_code/requirements.txt", "w") as f:
        f.write(requirements)

    # Package
    with tarfile.open("stock_training_code.tar.gz", "w:gz") as tar:
        tar.add("stock_training_code", arcname=".")

    print("✅ Training code packaged")
    return "stock_training_code.tar.gz"


def upload_training_code(bucket_name, package_file):
    """Upload training code to S3"""
    print("📤 Uploading training code...")

    s3 = boto3.client('s3', region_name=AWS_REGION)

    s3_key = "training-code/stock_training_code.tar.gz"
    s3.upload_file(package_file, bucket_name, s3_key)

    s3_uri = f"s3://{bucket_name}/{s3_key}"
    print(f"✅ Training code uploaded: {s3_uri}")

    return s3_uri


def create_sagemaker_training_job(bucket_name, training_code_uri):
    """Create SageMaker training job"""
    print("🚀 Creating SageMaker training job...")

    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)

    timestamp = int(time.time())
    job_name = f"stock-prediction-{timestamp}"

    # Use SKLearn container
    image_uri = "246618743249.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3"

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
                    'S3Uri': f"s3://{bucket_name}/training-data/",
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'text/csv'
        }],
        'OutputDataConfig': {
            'S3OutputPath': f"s3://{bucket_name}/model-output/"
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 20
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 1800  # 30 minutes
        },
        'HyperParameters': {
            'sagemaker_program': 'train.py',
            'sagemaker_submit_directory': training_code_uri
        }
    }

    try:
        response = sagemaker.create_training_job(**config)
        print(f"✅ Training job created: {job_name}")
        return job_name

    except Exception as e:
        print(f"❌ Training job failed: {e}")
        return None


def monitor_training_job(job_name):
    """Monitor training progress"""
    print(f"👀 Monitoring training job: {job_name}")

    sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)

    while True:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']

            print(f"Status: {status}")

            if status == 'Completed':
                print("🎉 Training completed!")
                return response['ModelArtifacts']['S3ModelArtifacts']
            elif status == 'Failed':
                print("❌ Training failed!")
                return None
            elif status == 'Stopped':
                print("⏹️ Training stopped")
                return None

            time.sleep(30)  # Check every 30 seconds

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)


def main():
    """Main project workflow"""
    print("🎓 UNIVERSITY STOCK PREDICTION ML PROJECT")
    print("=" * 60)

    # Step 1: Get stock data
    df = fetch_stock_data()
    if df is None:
        return

    # Step 2: Prepare ML data
    ml_data = prepare_ml_data(df)

    # Step 3: Create S3 bucket
    bucket_name = create_s3_bucket()
    if bucket_name is None:
        return

    # Step 4: Upload data
    data_uri = upload_data_to_s3(bucket_name)
    if data_uri is None:
        return

    # Step 5: Create training script
    create_training_script()

    # Step 6: Package and upload training code
    package_file = package_training_code()
    training_code_uri = upload_training_code(bucket_name, package_file)

    # Step 7: Start training job
    job_name = create_sagemaker_training_job(bucket_name, training_code_uri)
    if job_name is None:
        return

    # Step 8: Monitor training
    model_artifacts = monitor_training_job(job_name)

    if model_artifacts:
        print("\n" + "=" * 60)
        print("🎉 STOCK PREDICTION ML PROJECT COMPLETE!")
        print("=" * 60)
        print(f"Training Job: {job_name}")
        print(f"Model Artifacts: {model_artifacts}")
        print(f"S3 Bucket: {bucket_name}")
        print("Project Type: Stock Price Prediction")
        print("Algorithm: Random Forest Regression")
        print("Data: Apple (AAPL) Stock Prices")
        print("Perfect for university presentation! 📊")
        print("=" * 60)
    else:
        print("❌ Project failed")


if __name__ == "__main__":
    main()