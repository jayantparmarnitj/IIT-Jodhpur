from dataclasses import dataclass, field
import pandas as pd

@dataclass
class PaperBroker:
    initial_cash: float = 100000.0
    cash: float = field(init=False)
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_cash

    def buy(self, symbol: str, qty: int, price: float, date: pd.Timestamp):
        cost = qty * price
        if self.cash < cost:
            print(f"Not enough cash to buy {qty} {symbol}")
            return
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, 0) + qty
        self.trades.append({'date': date, 'side': 'BUY', 'symbol': symbol, 'qty': qty, 'price': price})

    def sell(self, symbol: str, qty: int, price: float, date: pd.Timestamp):
        if self.positions.get(symbol, 0) < qty:
            print(f"Not enough {symbol} to sell")
            return
        self.cash += qty * price
        self.positions[symbol] -= qty
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        self.trades.append({'date': date, 'side': 'SELL', 'symbol': symbol, 'qty': qty, 'price': price})

    def get_portfolio_value(self, prices: dict):
        value = self.cash + sum(prices[s] * q for s, q in self.positions.items())
        return value
