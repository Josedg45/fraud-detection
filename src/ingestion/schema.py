# src/ingestion/schema.py
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check

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
        "label": Column(int, Check.isin([0, 1]))
    },
    strict=True
)
