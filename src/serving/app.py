from functools import lru_cache

from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc
from pydantic import BaseModel, Field

from src.serving.decision import fraud_decision


class PredictRequest(BaseModel):
    amount: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    txn_count_1h: int = Field(..., ge=0)
    txn_sum_24h: float = Field(..., ge=0)
    user_id: int
    device_id: int
    transaction_id: int


app = FastAPI(title="Fraud Detection API")


@lru_cache(maxsize=1)
def get_model():
    return mlflow.pyfunc.load_model("models:/fraud_lightgbm@staging")


@app.post("/predict")
def predict(payload: PredictRequest):
    df = pd.DataFrame([payload.model_dump()])
    score = float(get_model().predict(df)[0])
    decision = fraud_decision(score)

    return {
        "fraud_score": score,
        "decision": decision
    }
