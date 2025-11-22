import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))  # ensure src is importable

import yfinance as yf
from simulator.execution.broker import PaperBroker
from simulator.strategies.ma_crossover import moving_average_crossover_strategy
from simulator.strategies.btst_ml import ml_btst_strategy
from simulator.backtest.engine import run_backtest, evaluate_and_plot

SYMBOL = "TVSMOTOR.NS"
START_DATE = "2022-01-01"
END_DATE = "2025-08-30"
SHORT_WINDOW = 20
LONG_WINDOW = 50
INITIAL_CASH = 100_000.0

if __name__ == '__main__':
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True)
    if df.empty:
        print('No data downloaded. Exiting.')
        raise SystemExit(1)
    df = df.reset_index()
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    df.index = df['Date']

    # ---------------------------
    # 1. MA Crossover Strategy
    # ---------------------------
    broker = PaperBroker(initial_cash=INITIAL_CASH)
    portfolio_history, signals = run_backtest(
        data=df,
        strategy_func=moving_average_crossover_strategy,
        broker=broker,
        symbol=SYMBOL,
        short_window=SHORT_WINDOW,
        long_window=LONG_WINDOW
    )
    if not portfolio_history.empty:
        evaluate_and_plot(
            portfolio_history,
            signals,
            df,
            SYMBOL,
            INITIAL_CASH,
            broker=broker,
            strategy_name="MA Crossover"
        )
        portfolio_history.to_csv('portfolio_history_ma.csv')
        print('Saved portfolio_history_ma.csv')
    else:
        print('No trades executed for MA Crossover.')

    # ---------------------------
    # 2. ML BTST Strategies
    # ---------------------------
    for algo in ["linear", "naive_bayes", "knn"]:
        broker = PaperBroker(initial_cash=INITIAL_CASH)
        portfolio_history, signals = run_backtest(
            data=df,
            strategy_func=ml_btst_strategy,
            broker=broker,
            symbol=SYMBOL,
            algo=algo
        )
        if not portfolio_history.empty:
            evaluate_and_plot(
                portfolio_history,
                signals,
                df,
                SYMBOL,
                INITIAL_CASH,
                broker=broker,
                strategy_name=f"ML BTST ({algo})"
            )
            portfolio_history.to_csv(f'portfolio_history_{algo}.csv')
            print(f'Saved portfolio_history_{algo}.csv')
        else:
            print(f'No trades executed for {algo}.')
