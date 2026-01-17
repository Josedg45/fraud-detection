import pandas as pd
import os
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

def load_csv(path: str) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def compute_metrics(y_true, y_pred_proba, threshold=0.2):

    preds_binary = (y_pred_proba >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_pred_proba)
    pr = average_precision_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, preds_binary)
    report = classification_report(y_true, preds_binary)
    
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "threshold": threshold,
        "confusion_matrix": cm,
        "classification_report": report
    }

def save_metrics(metrics_dict, path="metrics.txt"):

    with open(path, "w") as f:
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v}\n")

def preprocess_payload(payload: dict):

    df = pd.DataFrame([payload])
    return df
