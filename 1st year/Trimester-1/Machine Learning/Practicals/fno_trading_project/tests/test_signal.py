
from fno_trading.option_utils import OptionUtils

def test_get_trade_levels():
    opt = {'strikePrice':22100,'type':'CE','lastPrice':120}
    levels = OptionUtils.get_trade_levels(opt, rr_ratio=3.0, sl_buffer=0.1, btst=False)
    assert levels['entry']==120
    assert len(levels['targets'])==3
