
from typing import Tuple
from loguru import logger

class RiskManager:
    def __init__(self, capital: float = 500000.0, risk_per_trade: float = 0.02, lot_size: int = 50):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.lot_size = lot_size

    def calculate_position(self, entry: float, stop_loss: float) -> Tuple[int, float]:
        """Return (lots, capital_required)"""
        risk_per_unit = entry - stop_loss
        if risk_per_unit <= 0:
            return 0, 0.0
        max_risk = self.capital * self.risk_per_trade
        units = int(max_risk // risk_per_unit)
        lots = units // self.lot_size
        capital_required = lots * self.lot_size * entry
        logger.debug(f"Risk calc: risk_per_unit={risk_per_unit}, units={units}, lots={lots}, capital_required={capital_required}")
        return lots, capital_required

    def check_before_submit(self, lots: int, capital_required: float) -> Tuple[bool,str]:
        if lots <= 0:
            return False, "Position size zero or insufficient risk budget"
        if capital_required > self.capital:
            return False, "Insufficient capital for required margin"
        return True, "OK"
