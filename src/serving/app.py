from functools import lru_cache
from typing import Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from src.config import FEATURES, MODEL_URI, PCA_FEATURES, TRACKING_URI
from src.serving.decision import fraud_decision


class _BaseRequest(BaseModel):
    transaction_id: Optional[str] = None
    amount: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)


# V1..V28: anonymized PCA components, as in the Kaggle dataset.
PredictRequest = create_model(
    "PredictRequest",
    __base__=_BaseRequest,
    **{name: (float, ...) for name in PCA_FEATURES},
)


app = FastAPI(title="Fraud Detection API")


@lru_cache(maxsize=1)
def get_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(MODEL_URI)


def to_features(payload) -> pd.DataFrame:
    """One-row frame with the columns in training order."""
    return pd.DataFrame([payload.model_dump()])[FEATURES].astype("float64")


@app.post("/predict")
def predict(payload: PredictRequest):
    score = float(get_model().predict(to_features(payload))[0])
    return {
        "transaction_id": payload.transaction_id,
        "fraud_score": score,
        "decision": fraud_decision(score),
    }
