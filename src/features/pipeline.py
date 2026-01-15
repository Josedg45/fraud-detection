from .temporal import add_temporal_features
from .aggregates import add_user_aggregates

def build_features(df):
    df = add_temporal_features(df)
    df = add_user_aggregates(df)
    return df