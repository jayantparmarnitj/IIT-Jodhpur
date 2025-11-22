
# src/simulator/strategies/candlestick.py

import pandas as pd
from src.simulator.features import patterns

def candlestick_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Candlestick pattern-based strategy.
    - Buy on Bullish Engulfing or Hammer.
    - Sell on Bearish Engulfing or Shooting Star.
    """

    signals = pd.DataFrame(index=data.index)
    signals["positions"] = 0.0  # default no position

    # Detect patterns
    signals["bullish_engulfing"] = patterns.detect_bullish_engulfing(data)
    signals["bearish_engulfing"] = patterns.detect_bearish_engulfing(data)
    signals["hammer"] = patterns.detect_hammer(data)
    signals["shooting_star"] = patterns.detect_shooting_star(data)

    # --- Trading Logic ---
    # Buy if Bullish Engulfing or Hammer
    buy_signals = signals["bullish_engulfing"] | signals["hammer"]

    # Sell if Bearish Engulfing or Shooting Star
    sell_signals = signals["bearish_engulfing"] | signals["shooting_star"]

    signals.loc[buy_signals, "positions"] = 1.0
    signals.loc[sell_signals, "positions"] = -1.0

    return signals
