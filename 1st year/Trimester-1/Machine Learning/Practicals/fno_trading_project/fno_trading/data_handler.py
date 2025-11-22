
import pandas as pd
from typing import Callable, Dict, Any, List
from .types import Tick
from loguru import logger

class DataHandler:
    """Receive ticks (from BrokerClient) and provide OHLC candles & subscription API."""
    def __init__(self):
        # in-memory store for demo; replace with Redis/Kafka for production
        self.ticks: Dict[str, List[Tick]] = {}

    def persist_tick(self, tick: Tick) -> None:
        self.ticks.setdefault(tick.symbol, []).append(tick)
        logger.debug(f"Persisted tick: {tick}")

    def get_latest_candles(self, symbol: str, timeframe: str = '1m', n: int = 200) -> pd.DataFrame:
        # builds simple OHLC from ticks for demo purposes
        arr = self.ticks.get(symbol, [])
        if not arr:
            return pd.DataFrame(columns=['open','high','low','close','volume','timestamp'])
        # for demo, convert last n ticks to 1-row candles per tick (not real aggregation)
        df = pd.DataFrame([{'open':t.price,'high':t.price,'low':t.price,'close':t.price,'volume':t.volume,'timestamp':t.timestamp} for t in arr[-n:]])
        return df
