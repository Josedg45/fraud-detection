import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "data/processed/features.csv"

def train():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="label")
    X = X.select_dtypes(include=["number"])
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        ))
    ])

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=500,
        class_weight="balanced"
    )


    stack = StackingClassifier(
        estimators=[
            ("logistic", logistic),
            ("lgbm", lgbm)
        ],
        final_estimator=LogisticRegression(),
        stack_method="predict_proba",
        n_jobs=-1
    )

    with mlflow.start_run(run_name="stacking_model"):
        stack.fit(X_train, y_train)

        preds = stack.predict_proba(X_val)[:, 1]

        roc = roc_auc_score(y_val, preds)
        pr = average_precision_score(y_val, preds)

        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("pr_auc", pr)

        mlflow.sklearn.log_model(
            stack,
            artifact_path="model",
            registered_model_name="fraud_stacking"
        )

        print(f"ROC-AUC: {roc:.4f}")
        print(f"PR-AUC: {pr:.4f}")

if __name__ == "__main__":
    train()
