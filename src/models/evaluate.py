import pandas as pd
from lightgbm import LGBMClassifier

from src.models.data import load_splits
from src.models.metrics import compute_metrics

THRESHOLD = 0.2


def evaluate():
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_splits()

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=10,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

    metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1], threshold=THRESHOLD)
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC:  {metrics['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["report"])


if __name__ == "__main__":
    evaluate()
