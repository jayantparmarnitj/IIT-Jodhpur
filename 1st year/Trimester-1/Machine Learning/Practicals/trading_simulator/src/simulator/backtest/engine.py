import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

def run_backtest(data, strategy_func, broker, symbol, **strategy_params):
    """
    Runs the backtest simulation.
    Ensures only one active position is held at a time (like MA crossover).
    Works for both ML and rule-based strategies.
    """
    signals = strategy_func(data, **strategy_params)
    portfolio_values = []

    print("--- Starting Backtest Simulation ---")
    for i in range(len(signals)):
        current_date = signals.index[i]
        signal_row = signals.iloc[i]

        # Next day's open price for execution
        if i + 1 < len(data):
            trade_price = float(data['Open'].iloc[i+1])
        else:
            continue  # Can't trade on the last day

        # ✅ Position management
        if signal_row['positions'] == 1.0:  # Buy signal
            if symbol not in broker.positions or broker.positions[symbol] == 0:
                broker.buy(symbol, qty=10, price=trade_price, date=current_date)

        elif signal_row['positions'] == -1.0:  # Sell signal
            if symbol in broker.positions and broker.positions[symbol] > 0:
                broker.sell(symbol, qty=broker.positions[symbol], price=trade_price, date=current_date)

        # Record daily portfolio value
        current_price = float(data['Close'].iloc[i])
        current_prices = {symbol: current_price}
        portfolio_values.append({
            'date': current_date,
            'portfolio_value': broker.get_portfolio_value(current_prices)
        })

    print("--- Simulation Complete ---")
    return pd.DataFrame(portfolio_values).set_index('date'), signals


import numpy as np
import matplotlib.pyplot as plt

def evaluate_and_plot(portfolio_values, signals, data, symbol, initial_cash, broker=None, strategy_name="Strategy"):
    """Calculates performance metrics and plots the results."""
    # --- Performance Metrics ---
    final_value = portfolio_values['portfolio_value'].iloc[-1]
    returns = portfolio_values['portfolio_value'].pct_change().dropna()
    total_return = (final_value / initial_cash) - 1
    
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = 0.0 if std_return == 0 else mean_return / std_return * np.sqrt(252)

    print(f"\n--- {strategy_name} Performance ---")
    print(f"Initial Portfolio Value: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Total Return:            {total_return:.2%}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Always plot Close price
    ax1.plot(data.index, data['Close'], label='Close Price', color='royalblue', alpha=0.7)

    # ✅ Plot moving averages only if they exist (for MA crossover)
    if 'short_ma' in signals.columns and 'long_ma' in signals.columns:
        ax1.plot(signals.index, signals['short_ma'], label='Short MA', color='orange', linestyle='--')
        ax1.plot(signals.index, signals['long_ma'], label='Long MA', color='purple', linestyle='--')

    ax1.set_title(f"{strategy_name} Simulation for {symbol}")
    ax1.set_ylabel("Stock Price (INR)")
    ax1.legend(loc='upper left')

    # ✅ Plot only executed trades from broker
    if broker and hasattr(broker, "trades"):
        for trade in broker.trades:
            if trade["side"] == "BUY":
                ax1.scatter(trade["date"], trade["price"],
                            marker="^", color="lime", s=120,
                            label="Buy" if "Buy" not in ax1.get_legend_handles_labels()[1] else "",
                            edgecolors="black", zorder=5)
            elif trade["side"] == "SELL":
                ax1.scatter(trade["date"], trade["price"],
                            marker="v", color="red", s=120,
                            label="Sell" if "Sell" not in ax1.get_legend_handles_labels()[1] else "",
                            edgecolors="black", zorder=5)

    # Portfolio Value on a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(portfolio_values.index, portfolio_values['portfolio_value'],
             label='Portfolio Value', color='green', linewidth=2)
    ax2.set_ylabel("Portfolio Value ($)", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.tight_layout()
    plt.show()
