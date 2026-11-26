"""Shared settings for training, serving and monitoring."""
import os

# Training registers the model here, and serving/monitoring load it from here.
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

MODEL_NAME = "fraud_lightgbm"
MODEL_ALIAS = "staging"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# PCA components released with the Kaggle dataset. They carry most of the signal.
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

# Columns the model is trained on, in order. LightGBM predicts by position,
# so every caller must send columns in this order.
#
# The user/device/merchant columns are simulated in load_kaggle.py (the Kaggle
# data has no such fields), so IDs and per-user aggregates are NOT model features:
# they would only add noise. They stay in the pipeline to show the feature design
# that would apply to real transaction data.
FEATURES = PCA_FEATURES + ["amount", "hour"]

# Chronological split: train on the oldest data, early-stop on the next slice,
# report metrics on the most recent slice (closer to how fraud models are deployed).
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15
