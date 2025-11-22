
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict, List
import time, json
from .config_manager import ConfigManager
from .logger import setup_logging
from .broker_client import BrokerClient
from .strategy_engine import OptionBreakoutStrategy, Signal as StrategySignal
from .option_utils import OptionUtils
from .data_handler import DataHandler

app = FastAPI(title="MCP FNO Server")
logger = setup_logging()

class JsonRpcRequest(BaseModel):
    jsonrpc: str
    method: str
    params: Dict[str, Any] = {}
    id: Any = None

@app.post('/rpc')
async def rpc(body: JsonRpcRequest):
    method = body.method
    params = body.params or {}
    req_id = body.id
    try:
        if method == 'initialize':
            result = {'capabilities':{'tools':['get_signals','list_strategies']}, 'time': time.time()}
        elif method == 'get_signals':
            # Expects params: underlying, underlying_price, option_chain (list)
            underlying = params.get('underlying','NIFTY')
            underlying_price = float(params.get('underlying_price', 22000))
            option_chain = params.get('option_chain', [])
            # Build strategy and simulate a tick
            strat = OptionBreakoutStrategy(option_chain, underlying_price)
            # create a dummy tick (not used heavily)
            from .types import Tick
            t = Tick(symbol=underlying, price=underlying_price, volume=1000)
            signal = strat.on_tick(t)
            if signal:
                result = {'signals':[signal.as_dict()]}
            else:
                result = {'signals':[]}
        elif method == 'list_strategies':
            result = {'strategies':['OptionBreakout-OI']}
        else:
            raise ValueError('unknown method')
    except Exception as e:
        return {'jsonrpc':'2.0', 'id': req_id, 'error':{'code':-32000,'message':str(e)}}
    return {'jsonrpc':'2.0', 'id': req_id, 'result': result}
