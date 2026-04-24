# IMPLEMENT.md - Live Trading Implementation Log

## Task Summary

- Task: Build live trading system from SOTA backtest
- Date: 2026-04-25
- Agent: Sisyphus

## Data and Runtime

- Data root used: `/mnt/data/finance/cryptocurrency/binance/BTCUSDT/`
- Hive partition verified: ✅ 1h candles available
- Compute used: CPU (XGBoost), GPU available (RTX 3090)
- Engine choice: Polars (all feature extraction), DuckDB (historical joins)

## Changes Applied

- File: `live_trading/src/config.py`
  - Why: Lock SOTA parameters
  - What changed: Froze XGBoost params, barrier config, position sizing

- File: `live_trading/src/engine.py`
  - Why: Core trading loop
  - What changed: Async tick-based engine with signal → order flow

- File: `live_trading/src/features.py`
  - Why: Match backtest feature extraction exactly
  - What changed: 31-feature extraction matching `src/liquidation_map/ml/features.py`

- File: `live_trading/src/risk_manager.py`
  - Why: Position and risk management
  - What changed: Triple barrier, daily/weekly limits, position tracking

- File: `live_trading/src/executor.py`
  - Why: Binance API integration
  - What changed: HMAC auth, order placement, position sync

## Strategy Track Results

- Baseline summary: Simple momentum - underperformed
- Parametric summary: 8 configs tested, Long Horizon (48h) best
- ML/DL summary: XGBoost (Sharpe 5.19) > Hybrid CNN-LSTM (Sharpe 2.3)
- Cross-track comparison: See `docs/benchmark.html`
- Profitable candidate selected: Optuna-Optimized XGBoost

## Metrics Table (SOTA)

| Metric | Value |
|--------|-------|
| Return | +2.82% |
| Alpha | +19.95% |
| Sharpe | 5.19 |
| Max DD | 0.89% |
| Win Rate | 58.3% |
| Profit Factor | 2.56 |
| Trades | 24 |
| Expected return/trade | 0.12% |

## ML/DL Split Evidence

- Train period: 2025-06-01 ~ 2025-12-31 (7 months)
- Test period (latest): 2026-01-01 ~ 2026-04-22 (111 days)

## Decisions

- Decision: Use binary classification (long vs short) instead of 3-class
  - Alternatives considered: 3-class (buy/hold/sell)
  - Reason chosen: Better signal clarity, avoid noisy hold predictions

- Decision: No leverage for live trading
  - Alternatives considered: 2x-5x leverage
  - Reason chosen: Risk management, match backtest assumptions

## Verification

- Commands run:
  - Command: `python live_trading/tests/test_risk.py`
  - Result: ✅ ALL RISK TESTS PASSED

  - Command: `python live_trading/tests/test_parity.py`
  - Result: ⏳ Skipped (no backtest data yet - needs historical run)

- Manual checks:
  - Check: Model loads correctly
  - Result: ✅ `xgb_optuna_best.json` loads, predictions work

## Publishing

- `docs/` updated: ✅ benchmark.html with SOTA results
- `pages/` updated: ✅ Navigation links added

## Practical Trading Simulation (Pages)

- Seed capital: $10,000 USDT
- Position size per trade: 10% ($1,000)
- Execution model: Market orders (no limit orders)
- Fees: 0.04% taker fee
- Slippage: 5 bps assumed
- Leverage: 1x (no leverage)

## Deviations from Plan

- Deviation: Skipped full parity test
  - Reason: Need to run historical simulation first
  - Impact: CP-001 pending verification

## Open Questions

- Paper trading duration: 7 days sufficient?
- Alert system: Slack vs Email vs both?

## Next Steps

1. Run historical simulation to generate parity test data
2. Complete Svelte dashboard (Dashboard/History/Settings)
3. Start 7-day paper trading period
4. Complete CP-001 through CP-005

## Live Trading Readiness (Go Scope)

- Strategy note path: `live_trading/docs/DEVELOPMENT.md`
- Checkpoint list path: `live_trading/checkpoints/VERIFICATION.md`
- Validation completion status: 1/5 checkpoints passed (CP-004)

## Frontend Baseline (Svelte Scope)

- Dashboard tab: Pending implementation
- History tab: Pending implementation
- Settings tab: Pending implementation
- API key/secret key/on-off fields confirmed: ✅ In YAML configs
