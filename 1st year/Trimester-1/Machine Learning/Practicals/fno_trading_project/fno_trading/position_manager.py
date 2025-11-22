
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float
    side: str  # BUY/SELL

class PositionManager:
    def __init__(self):
        self.positions: Dict[str, Position] = {}

    def update_on_fill(self, fill: Dict[str, Any]) -> None:
        # fill = {symbol, qty, avg_price, side}
        sym = fill['symbol']
        if sym in self.positions:
            pos = self.positions[sym]
            # simplistic update: replaces position
            pos.qty = fill['qty']
            pos.avg_price = fill['avg_price']
            pos.side = fill['side']
        else:
            self.positions[sym] = Position(symbol=sym, qty=fill['qty'], avg_price=fill['avg_price'], side=fill['side'])

    def get_positions(self):
        return list(self.positions.values())

    def compute_unrealized_pnl(self, current_prices: Dict[str,float]) -> float:
        pnl = 0.0
        for sym, pos in self.positions.items():
            cp = current_prices.get(sym, pos.avg_price)
            if pos.side.upper() == 'BUY':
                pnl += (cp - pos.avg_price) * pos.qty
            else:
                pnl += (pos.avg_price - cp) * pos.qty
        return pnl
