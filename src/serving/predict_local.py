import mlflow
import mlflow.pyfunc
import pandas as pd

from src.config import FEATURES, MODEL_URI, TRACKING_URI


def run():
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)

    sample = pd.DataFrame([{
        "transaction_id": 999,
        "user_id": 102,
        "amount": 250.0,
        "device_id": 5,
        "hour": 22,
        "day_of_week": 5,
        "txn_count_1h": 7,
        "txn_sum_24h": 12,
    }])[FEATURES].astype("float64")

    score = model.predict(sample)[0]
    print("Fraud score:", score)


if __name__ == "__main__":
    run()
