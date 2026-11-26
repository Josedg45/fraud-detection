import pandas as pd
import pytest

from src.serving.app import PredictRequest, predict, get_model


class DummyModel:
    def predict(self, df: pd.DataFrame):
        assert len(df) == 1
        return [0.42]


def test_predict_request_valid_ranges():
    payload = PredictRequest(
        amount=10.0,
        hour=12,
        day_of_week=3,
        txn_count_1h=2,
        txn_sum_24h=100.0,
        user_id=1,
        device_id=2,
        transaction_id=3,
    )
    assert payload.hour == 12


def test_predict_request_invalid_hour():
    with pytest.raises(Exception):
        PredictRequest(
            amount=10.0,
            hour=24,
            day_of_week=3,
            txn_count_1h=2,
            txn_sum_24h=100.0,
            user_id=1,
            device_id=2,
            transaction_id=3,
        )


def test_predict_uses_model(monkeypatch):
    get_model.cache_clear()
    monkeypatch.setattr("src.serving.app.mlflow.pyfunc.load_model", lambda _: DummyModel())

    payload = PredictRequest(
        amount=10.0,
        hour=12,
        day_of_week=3,
        txn_count_1h=2,
        txn_sum_24h=100.0,
        user_id=1,
        device_id=2,
        transaction_id=3,
    )

    result = predict(payload)
    assert result["fraud_score"] == 0.42
    assert result["decision"] in {"ALLOW", "CHALLENGE", "BLOCK"}


def test_predict_sends_columns_in_training_order(monkeypatch):
    from src.config import FEATURES

    seen = {}

    class RecordingModel:
        def predict(self, df):
            seen["columns"] = list(df.columns)
            return [0.1]

    get_model.cache_clear()
    monkeypatch.setattr("src.serving.app.mlflow.pyfunc.load_model", lambda _: RecordingModel())

    predict(PredictRequest(
        amount=10.0, hour=12, day_of_week=3, txn_count_1h=2,
        txn_sum_24h=100.0, user_id=1, device_id=2, transaction_id=3,
    ))
    assert seen["columns"] == FEATURES
