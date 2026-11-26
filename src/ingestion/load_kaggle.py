import numpy as np
import pandas as pd

from src.config import PCA_FEATURES
from .validate import validate_transactions

# The Kaggle data covers two days in September 2013; "Time" is seconds since the
# first transaction. The exact start date is not published, so this is an anchor.
START_TS = pd.Timestamp("2013-09-01")


def load_data(path: str, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    rng = np.random.default_rng(seed)

    df = df.rename(columns={"Amount": "amount", "Class": "label"})
    df["transaction_id"] = df.index.astype(str)
    df["ts"] = START_TS + pd.to_timedelta(df["Time"], unit="s")

    # Simulated context fields (not in the Kaggle data) to exercise the schema
    # and the per-user feature design. They are not used as model features.
    df["user_id"] = rng.integers(1, 5000, size=len(df)).astype(str)
    df["merchant"] = rng.choice(["amazon", "walmart", "uber", "netflix"], size=len(df))
    df["country"] = rng.choice(["US", "CO", "MX", "BR"], size=len(df))
    df["device_id"] = rng.integers(1, 10000, size=len(df)).astype(str)
    df["payment_method"] = rng.choice(["credit", "debit"], size=len(df))

    cols = [
        "transaction_id", "ts", "user_id", "amount",
        "merchant", "country", "device_id", "payment_method",
        *PCA_FEATURES,
        "label",
    ]
    return validate_transactions(df[cols])


if __name__ == "__main__":
    print(load_data("data/raw/creditcard.csv").head())
