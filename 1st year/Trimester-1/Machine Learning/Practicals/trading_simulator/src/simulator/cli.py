import pandas as pd
from .data.loader import load_data
from .execution.broker import PaperBroker
from .strategies.ma_crossover import moving_average_crossover_strategy
from .backtest.engine import run_backtest
from .metrics.evaluator import evaluate

SYMBOL = 'TVSMOTOR.NS'
START, END = '2022-01-01', '2025-08-30'

if __name__ == '__main__':
    data = load_data(SYMBOL, START, END)
    broker = PaperBroker()
    signals, portfolio = run_backtest(data, moving_average_crossover_strategy, broker, SYMBOL, short_window=20, long_window=50)
    import pandas as pd
    portfolio_df = pd.DataFrame(portfolio).set_index('date')
    print(evaluate(portfolio_df, broker.initial_cash))
