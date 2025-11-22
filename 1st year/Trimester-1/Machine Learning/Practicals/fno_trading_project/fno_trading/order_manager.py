
from typing import Dict, Any, Tuple
from .broker_client import BrokerClient
from .types import OrderReceipt
from loguru import logger

class OrderManager:
    def __init__(self, broker: BrokerClient):
        self.broker = broker
        self.open_orders = {}

    def submit_order(self, symbol: str, qty: int, price: float, side: str, order_type: str = 'LIMIT', stop_loss: float = None) -> OrderReceipt:
        order = {'symbol': symbol, 'qty': qty, 'price': price, 'side': side, 'type': order_type, 'stop_loss': stop_loss}
        receipt = self.broker.place_order(order)
        self.open_orders[receipt.order_id] = receipt
        logger.info(f"Submitted order {receipt.order_id} for {symbol} qty={qty} price={price}")
        return receipt

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        resp = self.broker.cancel_order(order_id)
        if order_id in self.open_orders:
            self.open_orders.pop(order_id)
        logger.info(f"Cancelled order {order_id}")
        return resp
