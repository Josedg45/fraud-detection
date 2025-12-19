import pandas as pd
from pandera.errors import SchemaError
from .schema import transaction_schema

def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    try:
        validated_df = transaction_schema.validate(df)
        return validated_df
    except SchemaError as e:
        print("❌ Error de validación de esquema")
        raise e