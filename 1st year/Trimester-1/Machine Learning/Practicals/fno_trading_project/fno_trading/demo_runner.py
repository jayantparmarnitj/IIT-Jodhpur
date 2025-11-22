
import random, json
from .config_manager import ConfigManager
from .logger import setup_logging
from .broker_client import BrokerClient
from .data_handler import DataHandler
from .strategy_engine import OptionBreakoutStrategy
from .option_utils import OptionUtils
from .risk_manager import RiskManager
from .order_manager import OrderManager

logger = setup_logging()

def generate_dummy_chain(spot=22050):
    strikes = list(range(int(spot)-500, int(spot)+501, 100))
    chain = []
    for s in strikes:
        # random ltp around distance from spot
        dist = abs(s - spot)
        base = max(20, 300 - (dist/10))
        ltp = round(max(5, random.gauss(base, base*0.1)),2)
        chain.append({'strikePrice':s, 'type':'CE', 'lastPrice':ltp, 'openInterest': random.randint(10000,200000), 'bidPrice':ltp-0.5,'askPrice':ltp+0.5, 'expiry':'2025-09-25'})
        chain.append({'strikePrice':s, 'type':'PE', 'lastPrice':ltp, 'openInterest': random.randint(10000,200000), 'bidPrice':ltp-0.5,'askPrice':ltp+0.5, 'expiry':'2025-09-25'})
    return chain

def run_demo():
    cfg = ConfigManager('config/config.json')
    capital = cfg.get('account.capital', 500000)
    risk_per_trade = cfg.get('account.risk_per_trade', 0.02)
    lot_size = cfg.get('trading.nifty_lot_size', 50)

    # create components
    dh = DataHandler()
    broker = BrokerClient(api_key=cfg.get('broker.api_key'), api_secret=cfg.get('broker.api_secret'))
    rm = RiskManager(capital=capital, risk_per_trade=risk_per_trade, lot_size=lot_size)
    om = OrderManager(broker=broker)

    # generate chain and strategy
    chain = generate_dummy_chain(22050)
    strat = OptionBreakoutStrategy(chain, underlying_price=22050)

    # simulate a tick
    from .types import Tick
    t = Tick(symbol='NIFTY', price=22050, volume=1000)
    signal = strat.on_tick(t)
    print('Generated Signal:', signal.as_dict() if signal else 'None')

    # calculate size
    if signal:
        lots, capital_req = rm.calculate_position(signal.entry, signal.stop_loss)
        approved, reason = rm.check_before_submit(lots, capital_req)
        print('Risk calc -> lots:', lots, 'capital_req:', capital_req, 'approved:', approved, 'reason:', reason)
        if approved and lots>0:
            # submit order (symbol use a placeholder contract name)
            receipt = om.submit_order(symbol=f"NIFTY{signal.strike}{signal.option_type}", qty=lots*signal.lot_size, price=signal.entry, side='BUY', order_type='LIMIT', stop_loss=signal.stop_loss)
            print('Order Receipt:', receipt)
        else:
            print('Order not submitted due to risk check')
    else:
        print('No signal')

if __name__ == '__main__':
    run_demo()
