def add_user_aggregates(df):

    df = df.set_index('ts').sort_index()

    df['txn_count_1h'] = (
        df.groupby('user_id')['label']
          .rolling('1h')
          .count()
          .reset_index(level = 0,drop = True)
    )

    df['txn_sum_24h'] = (
        df.groupby('user_id')['label']
          .rolling('24h')
          .count()
          .reset_index(level = 0,drop = True)
    )

    return df.reset_index().sort_values("ts")