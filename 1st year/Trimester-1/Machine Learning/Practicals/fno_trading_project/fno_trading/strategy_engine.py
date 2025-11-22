
from typing import List, Dict, Any, Optional
from .types import Tick
from .option_utils import OptionUtils
from .types import Order
from .types import OrderReceipt
from .types import Order as OrderType
from .types import OrderReceipt as OR
from loguru import logger
from .types import Order as OrderDataclass
from dataclasses import asdict
from .types import Tick

# Signal dataclass embedded to avoid circular imports
from dataclasses import dataclass, field
from typing import List as TList
import datetime

@dataclass
class Signal:
    symbol: str
    strike: int
    option_type: str
    expiry: str
    entry: float
    stop_loss: float
    targets: TList[float]
    btst: bool = False
    strategy: str = "default"
    win_rate: float = 0.0
    rr_score: float = 0.0
    profit_factor: float = 0.0
    lot_size: int = 50
    qty: int = 0
    capital_required: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def as_dict(self):
        return {k: (round(v,2) if isinstance(v,float) else v) for k,v in self.__dict__.items()}

class BaseStrategy:
    def on_tick(self, tick: Tick) -> Optional[Signal]:
        raise NotImplementedError

class OptionBreakoutStrategy(BaseStrategy):
    """Very simple example: pick best CE/PE by OI and propose long option buy."""
    def __init__(self, option_chain: List[Dict[str,Any]], underlying_price: float):
        self.chain = option_chain
        self.underlying = underlying_price

    def on_tick(self, tick: Tick) -> Optional[Signal]:
        # For demo, pick CE ATM by OI
        best = OptionUtils.get_best_strike_price(self.chain, self.underlying, option_type="CE", method="OI")
        if not best:
            return None
        levels = OptionUtils.get_trade_levels(best, rr_ratio=3.0, sl_buffer=0.1, btst=False)
        # simple heuristics for win_rate
        win_rate = 60.0 + min(30.0, (best.get('openInterest',0)/1000000.0)*100)
        # rr_score based on average target
        risk = levels['entry'] - levels['stop_loss']
        avg_reward = sum([t-levels['entry'] for t in levels['targets']]) / len(levels['targets'])
        rr_score = (avg_reward / risk) if risk>0 else 0.0
        profit_factor = win_rate / (100.0 - win_rate) if win_rate < 100 else 10.0
        signal = Signal(
            symbol="NIFTY",
            strike=int(best['strikePrice']),
            option_type=best['type'],
            expiry=best.get('expiry','UNKNOWN'),
            entry=levels['entry'],
            stop_loss=levels['stop_loss'],
            targets=levels['targets'],
            btst=False,
            strategy="OptionBreakout-OI",
            win_rate=round(win_rate,2),
            rr_score=round(rr_score,2),
            profit_factor=round(profit_factor,2)
        )
        return signal
