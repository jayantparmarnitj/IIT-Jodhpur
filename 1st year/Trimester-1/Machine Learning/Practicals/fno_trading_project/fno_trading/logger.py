
from loguru import logger
import sys

def setup_logging(logfile: str = "fno_trading.log"):
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(logfile, rotation="10 MB", retention="10 days", level="DEBUG")
    return logger
