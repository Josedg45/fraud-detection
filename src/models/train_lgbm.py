import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_PATH = "data/processed/features.csv"

def train():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")


    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="label")
    X = X.select_dtypes(include=["number"])
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

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraud_lightgbm"
        )

        print(f"ROC-AUC: {roc:.4f}")
        print(f"PR-AUC: {pr:.4f}")

if __name__ == "__main__":
    train()
