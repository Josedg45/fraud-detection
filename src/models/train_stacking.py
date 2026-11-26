import lightgbm as lgb
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import TRACKING_URI
from src.models.data import load_splits


def train():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("fraud-detection")

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_splits()
    # No early stopping here, so train on train + valid and report on test.
    X_fit = pd.concat([X_train, X_valid])
    y_fit = pd.concat([y_train, y_valid])

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    lgbm = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=500,
        scale_pos_weight=10,
        random_state=42,
        verbosity=-1,
    )
    stack = StackingClassifier(
        estimators=[("logistic", logistic), ("lgbm", lgbm)],
        final_estimator=LogisticRegression(),
        stack_method="predict_proba",
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="stacking_model"):
        stack.fit(X_fit, y_fit)
        preds = stack.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, preds)
        pr = average_precision_score(y_test, preds)
        mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr})
        mlflow.sklearn.log_model(stack, artifact_path="model", registered_model_name="fraud_stacking")

        print(f"Test ROC-AUC: {roc:.4f}")
        print(f"Test PR-AUC:  {pr:.4f}")


if __name__ == "__main__":
    train()
