import pandas as pd
import numpy as np

from .validate import validate_transactions


def load_data(path: str, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    rng = np.random.default_rng(seed)

    # Kaggle → esquema realista
    df["transaction_id"] = df.index.astype(str)
    df["ts"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(df.index, unit="s")
    df["user_id"] = rng.integers(1, 5000, size=len(df)).astype(str)
    df["merchant"] = rng.choice(["amazon", "walmart", "uber", "netflix"], size=len(df))
    df["country"] = rng.choice(["US", "CO", "MX", "BR"], size=len(df))
    df["device_id"] = rng.integers(1, 10000, size=len(df)).astype(str)
    df["payment_method"] = rng.choice(["credit", "debit"], size=len(df))

    df.rename(columns={"Amount": "amount", "Class": "label"}, inplace=True)

    cols = [
        "transaction_id", "ts", "user_id", "amount",
        "merchant", "country", "device_id", "payment_method", "label"
    ]

    df = df[cols]

    return validate_transactions(df)


if __name__ == "__main__":
    df = load_data("data/raw/creditcard.csv")
    print(df.head())
