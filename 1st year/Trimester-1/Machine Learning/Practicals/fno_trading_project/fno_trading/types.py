
from dataclasses import dataclass, field
from typing import List, Optional
import datetime

@dataclass
class Tick:
    symbol: str
    price: float
    volume: float
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class Order:
    symbol: str
    qty: int
    price: Optional[float]
    side: str  # BUY / SELL
    order_type: str  # MARKET / LIMIT / SL-M
    stop_loss: Optional[float] = None
    target: Optional[float] = None

@dataclass
class OrderReceipt:
    order_id: str
    status: str
    filled_qty: int = 0
    avg_price: Optional[float] = None
    broker_msg: str = ""
