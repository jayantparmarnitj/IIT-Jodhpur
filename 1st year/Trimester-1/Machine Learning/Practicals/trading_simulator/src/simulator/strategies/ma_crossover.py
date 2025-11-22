import pandas as pd
import numpy as np

def moving_average_crossover_strategy(data, short_window=20, long_window=50):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['short_ma'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_ma'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    signals.loc[signals.index[short_window:], 'signal'] = np.where(
        signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1.0, 0.0
    )
    signals['positions'] = signals['signal'].diff()
    return signals
