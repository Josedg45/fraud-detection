"""Shared settings for training, serving and monitoring."""
import os

# Training registers the model here, and serving/monitoring load it from here.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

MODEL_NAME = "fraud_lightgbm"
MODEL_ALIAS = "staging"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# Column order the model is trained on (numeric columns of data/processed/features.csv).
# LightGBM predicts by position, so every caller must send columns in this order.
FEATURES = [
    "transaction_id",
    "user_id",
    "amount",
    "device_id",
    "hour",
    "day_of_week",
    "txn_count_1h",
    "txn_sum_24h",
]
