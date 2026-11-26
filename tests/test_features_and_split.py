import pandas as pd

from src.config import FEATURES
from src.features.aggregates import add_user_aggregates
from src.models.data import time_split


def test_txn_sum_24h_sums_amounts_per_user():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2013-09-01 00:00", "2013-09-01 00:30", "2013-09-01 02:00", "2013-09-02 03:00"]),
        "user_id": ["a", "a", "b", "a"],
        "amount": [10.0, 5.0, 7.0, 1.0],
        "label": [0, 0, 0, 0],
    })
    out = add_user_aggregates(df).set_index("ts")

    assert out.loc["2013-09-01 00:30", "txn_sum_24h"] == 15.0   # 10 + 5
    assert out.loc["2013-09-01 00:30", "txn_count_1h"] == 2
    assert out.loc["2013-09-01 02:00", "txn_sum_24h"] == 7.0    # other user
    assert out.loc["2013-09-02 03:00", "txn_sum_24h"] == 1.0    # older ones fell out of the window


def test_time_split_is_chronological():
    df = pd.DataFrame({"ts": pd.date_range("2013-09-01", periods=100, freq="min")[::-1], "label": 0})
    train, valid, test = time_split(df)

    assert len(train) + len(valid) + len(test) == 100
    assert train["ts"].max() < valid["ts"].min()
    assert valid["ts"].max() < test["ts"].min()


def test_model_features_exclude_simulated_ids():
    for col in ["user_id", "device_id", "transaction_id", "txn_count_1h", "txn_sum_24h"]:
        assert col not in FEATURES


def test_user_aggregates_handle_duplicate_timestamps():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2013-09-01 00:00"] * 3 + ["2013-09-01 00:10"]),
        "user_id": ["a", "b", "a", "a"],
        "amount": [1.0, 100.0, 2.0, 4.0],
        "label": [0, 0, 0, 0],
    })
    out = add_user_aggregates(df)

    assert len(out) == 4
    last_a = out[(out.user_id == "a")].iloc[-1]
    assert last_a["txn_sum_24h"] == 7.0 and last_a["txn_count_1h"] == 3
    assert out[out.user_id == "b"]["txn_sum_24h"].tolist() == [100.0]
