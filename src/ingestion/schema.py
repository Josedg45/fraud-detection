# src/ingestion/schema.py
import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema

from src.config import PCA_FEATURES

transaction_schema = DataFrameSchema(
    {
        "transaction_id": Column(str, nullable=False),
        "ts": Column(pa.DateTime, nullable=False),
        "user_id": Column(str, nullable=False),
        "amount": Column(float, Check.ge(0)),
        "merchant": Column(str),
        "country": Column(str),
        "device_id": Column(str),
        "payment_method": Column(str),
        **{name: Column(float, nullable=False) for name in PCA_FEATURES},
        "label": Column(int, Check.isin([0, 1])),
    },
    strict=True,
)
