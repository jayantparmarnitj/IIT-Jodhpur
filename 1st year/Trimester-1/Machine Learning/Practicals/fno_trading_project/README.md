
# F&O Trading Starter (Modular)
This repository is a starter, modular architecture for a semi-automated Indian F&O trading system in Python.
It includes a clean design pattern (BrokerClient, DataHandler, StrategyEngine, RiskManager, OrderManager, PositionManager, OptionUtils, MCP server stub) and a runnable demo that uses simulated data.

## Quickstart (local demo)
1. Create and activate a Python 3.10+ venv:
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Edit `config/config.json` and fill broker credentials or keep placeholders for demo.
4. Run the demo script (uses simulated data):
```bash
python3 scripts/run_local_demo.py
```
5. (Optional) Run the MCP JSON-RPC server:
```bash
uvicorn fno_trading.mcp_server:app --reload --host 0.0.0.0 --port 4333
```
Then POST JSON-RPC requests to `http://localhost:4333/rpc` method=`get_signals` as described in the README sections.

## What is included
- `fno_trading/` : core modules
- `scripts/run_local_demo.py` : quick demo runner
- `config/config.json` : example config (use env-vars or Vault in prod)
- `requirements.txt` : libraries
- Simple MCP JSON-RPC endpoint (FastAPI) as `fno_trading.mcp_server`

## Notes & Next Steps
- Broker client is a stub. Replace methods in `fno_trading/broker_client.py` with real API calls (Zerodha/Upstox/Angel).
- Add credentials securely (env or vault) — do not commit secrets.
- Backtest strategies thoroughly before going live; use paper trading.
- Add tests and CI, and set up Prometheus/Grafana for metrics.
