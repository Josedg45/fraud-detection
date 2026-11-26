import numpy as np
import pandas as pd

from src.config import PCA_FEATURES
from src.ingestion import load_kaggle


def kaggle_frame(n=4):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Time": [0.0, 30.0, 3600.0, 90000.0][:n]})
    for name in PCA_FEATURES:
        df[name] = rng.normal(size=n)
    df["Amount"] = [1.0, 2.5, 3.0, 10.0][:n]
    df["Class"] = [0, 1, 0, 0][:n]
    return df


def test_load_data_keeps_pca_features_and_uses_time(monkeypatch):
    base_df = kaggle_frame()
    monkeypatch.setattr(load_kaggle.pd, "read_csv", lambda _: base_df.copy())

    df = load_kaggle.load_data("dummy.csv", seed=7)

    for name in PCA_FEATURES:
        assert name in df.columns
        np.testing.assert_allclose(df[name].to_numpy(), base_df[name].to_numpy())
    assert (df["ts"] - load_kaggle.START_TS).dt.total_seconds().tolist() == base_df["Time"].tolist()
    assert df["label"].tolist() == [0, 1, 0, 0]


def test_load_data_is_deterministic(monkeypatch):
    base_df = kaggle_frame()
    monkeypatch.setattr(load_kaggle.pd, "read_csv", lambda _: base_df.copy())

    assert load_kaggle.load_data("dummy.csv", seed=7).equals(load_kaggle.load_data("dummy.csv", seed=7))
