# src/simulator/features/patterns.py

import pandas as pd
import numpy as np

def detect_doji(data: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
    """
    Detect Doji candlestick pattern (open ≈ close).
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data with columns ["Open", "High", "Low", "Close"].
    threshold : float
        Relative difference threshold (default: 0.1%).
    
    Returns
    -------
    pd.Series
        Boolean Series where True indicates a Doji.
    """
    return (np.abs(data["Close"] - data["Open"]) <= threshold * (data["High"] - data["Low"]))



def detect_hammer(data: pd.DataFrame) -> pd.Series:
    """Detect Hammer candlestick pattern."""
    body = (data["Close"] - data["Open"]).abs()
    lower_shadow = (data[["Open", "Close"]].min(axis=1) - data["Low"]).abs()
    upper_shadow = (data["High"] - data[["Open", "Close"]].max(axis=1)).abs()

    # Ensure all are Series (avoid accidental DataFrame alignment issues)
    body = body.astype(float)
    lower_shadow = lower_shadow.astype(float)
    upper_shadow = upper_shadow.astype(float)

    return (lower_shadow > 2 * body) & (upper_shadow < body)



def detect_shooting_star(data: pd.DataFrame) -> pd.Series:
    """
    Detect Shooting Star (long upper shadow, small body, little lower shadow).
    """
    body = np.abs(data["Close"] - data["Open"])
    lower_shadow = data[["Open", "Close"]].min(axis=1) - data["Low"]
    upper_shadow = data["High"] - data[["Open", "Close"]].max(axis=1)
    
    return (upper_shadow > 2 * body) & (lower_shadow < body)


def detect_bullish_engulfing(data: pd.DataFrame) -> pd.Series:
    """
    Detect Bullish Engulfing pattern.
    """
    prev = data.shift(1)
    return (
        (prev["Close"] < prev["Open"]) &  # previous candle bearish
        (data["Close"] > data["Open"]) &  # current candle bullish
        (data["Close"] > prev["Open"]) & 
        (data["Open"] < prev["Close"])
    )


def detect_bearish_engulfing(data: pd.DataFrame) -> pd.Series:
    """
    Detect Bearish Engulfing pattern.
    """
    prev = data.shift(1)
    return (
        (prev["Close"] > prev["Open"]) &  # previous candle bullish
        (data["Close"] < data["Open"]) &  # current candle bearish
        (data["Close"] < prev["Open"]) & 
        (data["Open"] > prev["Close"])
    )
