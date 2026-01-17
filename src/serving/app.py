from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc
from src.serving.decision import fraud_decision

app = FastAPI(title="Fraud Detection API")

model = mlflow.pyfunc.load_model(
    "models:/fraud_lightgbm@staging"
)

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    score = float(model.predict(df)[0])
    decision = fraud_decision(score)

    return {
        "fraud_score": score,
        "decision": decision
    }
