# src/simulator/data/loader.py

import yfinance as yf
import pandas as pd

def load_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Load OHLCV data from Yahoo Finance with support for intraday intervals.
    
    Parameters
    ----------
    symbol : str
        Stock ticker (e.g., "TCS.NS", "AAPL").
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    interval : str
        Candle interval. Examples:
        - "1m"  : 1 minute  (last 7 days max)
        - "2m"  : 2 minutes (last 60 days max)
        - "5m"  : 5 minutes
        - "15m" : 15 minutes
        - "30m" : 30 minutes
        - "60m" : 1 hour
        - "1d"  : daily (default)
        - "1wk" : weekly
        - "1mo" : monthly
    
    Returns
    -------
    pd.DataFrame
        DataFrame with DateTime index and OHLCV columns.
    """
    
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True)
    
    if df.empty:
        print(f"⚠️ No data returned for {symbol} with interval={interval}")
        return df
    
    # Reset and clean index
    df = df.reset_index()
    if 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'Date'}, inplace=True)
    elif 'Date' not in df.columns:
        df.index.name = 'Date'
        df.reset_index(inplace=True)
    
    # Remove timezone (important for backtesting consistency)
    if pd.api.types.is_datetime64_any_dtype(df['Date']):
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
    
    df.set_index('Date', inplace=True)
    return df
