import numpy as np

def evaluate(portfolio_df, initial_cash):
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_cash) - 1
    returns = portfolio_df['value'].pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    return {'final_value': final_value, 'total_return': total_return, 'sharpe': sharpe}
