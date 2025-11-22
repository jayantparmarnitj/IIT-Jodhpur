def add_features(df):
    df['Return1'] = df['Close'].pct_change()
    df['Return5'] = df['Close'].pct_change(5)
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['Volatility10'] = df['Close'].pct_change().rolling(10).std()
    df = df.dropna().copy()
    return df
