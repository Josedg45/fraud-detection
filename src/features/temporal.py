def add_temporal_features(df):
    df = df.copy()
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek
    return df
