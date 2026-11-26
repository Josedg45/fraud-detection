"""Score the first transactions of the feature table with the registered model."""
import mlflow
import mlflow.pyfunc
import pandas as pd

from src.config import FEATURES, MODEL_URI, TRACKING_URI
from src.serving.decision import fraud_decision

DATA_PATH = "data/processed/features.csv"


def run(n: int = 5):
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)

    sample = pd.read_csv(DATA_PATH, nrows=n)
    scores = model.predict(sample[FEATURES].astype("float64"))
    for txn_id, label, score in zip(sample["transaction_id"], sample["label"], scores):
        print(f"txn {txn_id}: score={score:.4f} decision={fraud_decision(score)} label={label}")


if __name__ == "__main__":
    run()
