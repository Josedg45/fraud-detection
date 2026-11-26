import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import FEATURES, MODEL_ALIAS, MODEL_NAME, TRACKING_URI

DATA_PATH = "data/processed/features.csv"

def train():

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("fraud-detection")


    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].astype("float64")
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "verbosity": -1
    }

    with mlflow.start_run(run_name="lightgbm_fraud"):
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

        preds = model.predict(X_val)

        roc = roc_auc_score(y_val, preds)
        pr = average_precision_score(y_val, preds)

        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("pr_auc", pr)

        info = mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=infer_signature(X_val, preds),
        )

        # Serving and monitoring load models:/fraud_lightgbm@staging
        MlflowClient().set_registered_model_alias(
            MODEL_NAME, MODEL_ALIAS, info.registered_model_version
        )

        print(f"ROC-AUC: {roc:.4f}")
        print(f"PR-AUC: {pr:.4f}")

if __name__ == "__main__":
    train()
