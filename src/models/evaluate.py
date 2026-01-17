import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from src.models.metrics import compute_metrics

DATA_PATH = "data/processed/features.csv"
THRESHOLD = 0.2

def evaluate():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns="label").select_dtypes(include=["number"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_proba, threshold=THRESHOLD)

    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(metrics["report"])


if __name__ == "__main__":
    evaluate()