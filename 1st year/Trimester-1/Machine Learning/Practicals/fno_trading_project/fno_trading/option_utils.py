
from typing import List, Dict, Any, Optional

class OptionUtils:
    @staticmethod
    def get_best_strike_price(option_chain: List[Dict[str, Any]], underlying_price: float, option_type: str = "CE", method: str = "ATM") -> Optional[Dict[str, Any]]:
        """Select best strike from chain by method."""
        filtered = [o for o in option_chain if o.get('type') == option_type]
        if not filtered:
            return None
        if method == "ATM":
            best = min(filtered, key=lambda x: abs(x.get('strikePrice', 0) - underlying_price))
        elif method == "OI":
            best = max(filtered, key=lambda x: x.get('openInterest', 0))
        elif method == "Liquidity":
            best = min(filtered, key=lambda x: abs(x.get('askPrice', 1e9) - x.get('bidPrice', 0)))
        else:
            best = filtered[0]
        return best

    @staticmethod
    def get_trade_levels(option: Dict[str, Any], rr_ratio: float = 3.0, sl_buffer: float = 0.1, btst: bool = False) -> Dict[str, Any]:
        """Return entry, stop_loss and 3 targets (T1,T2,T3)."""
        entry = float(option.get('lastPrice', option.get('ltp', 0.0)))
        if entry <= 0:
            raise ValueError('Invalid option price for trade level calc')
        risk = entry * sl_buffer
        stop_loss = entry - risk
        # targets as 1R,2R,3R
        targets = [entry + (i * risk) for i in (1,2,3)]
        if btst:
            targets = [t * 1.1 for t in targets]  # widen for overnight
        return {'entry': round(entry,2), 'stop_loss': round(stop_loss,2), 'targets':[round(t,2) for t in targets]}
