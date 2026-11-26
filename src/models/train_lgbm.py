import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import MODEL_ALIAS, MODEL_NAME, TRACKING_URI
from src.models.data import load_splits


def train():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("fraud-detection")

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_splits()

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "scale_pos_weight": 10,
        "seed": 42,
        "verbosity": -1,
    }

    with mlflow.start_run(run_name="lightgbm_fraud"):
        model = lgb.train(
            params,
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)],
        )

        # Metrics on the most recent slice, never seen during training or early stopping.
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        roc = roc_auc_score(y_test, preds)
        pr = average_precision_score(y_test, preds)

        mlflow.log_params({**params, "best_iteration": model.best_iteration})
        mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr, "test_fraud_rate": float(y_test.mean())})

        info = mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=infer_signature(X_test, preds),
        )

        # Serving and monitoring load models:/fraud_lightgbm@staging
        MlflowClient().set_registered_model_alias(
            MODEL_NAME, MODEL_ALIAS, info.registered_model_version
        )

        print(f"Best iteration: {model.best_iteration}")
        print(f"Test ROC-AUC: {roc:.4f}")
        print(f"Test PR-AUC:  {pr:.4f}  (fraud rate in test: {y_test.mean():.4%})")


if __name__ == "__main__":
    train()
