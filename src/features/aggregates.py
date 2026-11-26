def add_user_aggregates(df):
    """Per-user rolling features: transaction count in the last hour and
    amount spent in the last 24 hours (both include the current transaction).

    Timestamps can repeat (they do in the Kaggle data), so results are aligned
    by position after sorting by user and time, not by the timestamp index.
    """
    df = df.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)
    by_user = df.set_index("ts").groupby("user_id", sort=False)["amount"]

    df["txn_count_1h"] = by_user.rolling("1h").count().to_numpy()
    df["txn_sum_24h"] = by_user.rolling("24h").sum().to_numpy()

    return df.sort_values("ts", kind="stable").reset_index(drop=True)
