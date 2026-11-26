import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.data import load_splits
from src.models.metrics import compute_metrics

THRESHOLD = 0.2


def train():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_splits()
    X_fit = pd.concat([X_train, X_valid])
    y_fit = pd.concat([y_train, y_valid])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_fit, y_fit)

    metrics = compute_metrics(y_test, pipeline.predict_proba(X_test)[:, 1], threshold=THRESHOLD)
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"\nThreshold: {THRESHOLD}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["report"])
    return pipeline


if __name__ == "__main__":
    train()
