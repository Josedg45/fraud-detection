"""Load the feature table and split it chronologically."""
import pandas as pd

from src.config import FEATURES, TEST_FRACTION, VALID_FRACTION

DATA_PATH = "data/processed/features.csv"


def time_split(df: pd.DataFrame):
    """Split into train / valid / test by timestamp, oldest to newest."""
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - TEST_FRACTION))
    valid_start = int(n * (1 - TEST_FRACTION - VALID_FRACTION))
    return df.iloc[:valid_start], df.iloc[valid_start:test_start], df.iloc[test_start:]


def xy(df: pd.DataFrame):
    return df[FEATURES].astype("float64"), df["label"]


def load_splits(path: str = DATA_PATH):
    df = pd.read_csv(path, parse_dates=["ts"])
    return [xy(part) for part in time_split(df)]
