import pandas as pd
import mlflow.pyfunc

MODEL_URI = "models:/fraud_lightgbm@staging"

def run():
    model = mlflow.pyfunc.load_model(MODEL_URI)

    sample = pd.DataFrame([{
        "amount": 250.0,
        "hour": 22,
        "day_of_week": 5,
        "user_tx_count_24h": 7,
        "user_avg_amount_24h": 90.5,
        "user_tx_sum_24h": 1200.0,
        "num_devices_30d": 2,
        "velocity_ip_changes": 1
    }])

    score = model.predict(sample)[0]
    print("Fraud score:", score)

if __name__ == "__main__":
    run()
