import pandas as pd

from src.ingestion import load_kaggle


def test_load_data_is_deterministic(monkeypatch):
    base_df = pd.DataFrame(
        {
            "Amount": [1.0, 2.5, 3.0],
            "Class": [0, 1, 0],
        }
    )

    monkeypatch.setattr(load_kaggle.pd, "read_csv", lambda _: base_df.copy())
    monkeypatch.setattr(load_kaggle, "validate_transactions", lambda df: df)

    first = load_kaggle.load_data("dummy.csv", seed=7)
    second = load_kaggle.load_data("dummy.csv", seed=7)

    assert first.equals(second)
    assert list(first.columns) == [
        "transaction_id",
        "ts",
        "user_id",
        "amount",
        "merchant",
        "country",
        "device_id",
        "payment_method",
        "label",
    ]
