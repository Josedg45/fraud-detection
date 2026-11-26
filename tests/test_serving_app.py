import pandas as pd
import pytest

from src.config import FEATURES, PCA_FEATURES
from src.serving.app import PredictRequest, get_model, predict


def payload(**overrides):
    data = {"transaction_id": "t1", "amount": 10.0, "hour": 12, **{v: 0.1 for v in PCA_FEATURES}}
    data.update(overrides)
    return PredictRequest(**data)


class DummyModel:
    def predict(self, df: pd.DataFrame):
        assert len(df) == 1
        return [0.42]


def test_predict_request_valid_ranges():
    assert payload().hour == 12


def test_predict_request_invalid_hour():
    with pytest.raises(Exception):
        payload(hour=24)


def test_predict_request_requires_pca_features():
    data = {"amount": 10.0, "hour": 12}
    with pytest.raises(Exception):
        PredictRequest(**data)


def test_predict_uses_model(monkeypatch):
    get_model.cache_clear()
    monkeypatch.setattr("src.serving.app.mlflow.pyfunc.load_model", lambda _: DummyModel())

    result = predict(payload())
    assert result["fraud_score"] == 0.42
    assert result["transaction_id"] == "t1"
    assert result["decision"] in {"ALLOW", "CHALLENGE", "BLOCK"}


def test_predict_sends_columns_in_training_order(monkeypatch):
    seen = {}

    class RecordingModel:
        def predict(self, df):
            seen["columns"] = list(df.columns)
            return [0.1]

    get_model.cache_clear()
    monkeypatch.setattr("src.serving.app.mlflow.pyfunc.load_model", lambda _: RecordingModel())

    predict(payload())
    assert seen["columns"] == FEATURES
