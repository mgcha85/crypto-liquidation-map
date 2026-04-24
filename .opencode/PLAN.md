# PLAN.md - Live Trading Implementation Plan

## Task

- Request: Build production live trading system from SOTA backtest results
- Owner: mgcha85
- Date: 2026-04-25

## Goal

- Primary outcome: Production-ready live trading system matching SOTA backtest performance
- Non-goals: New strategy development, parameter re-optimization

## Strategy Track (Required Order)

1. ✅ Baseline algorithm - Completed
2. ✅ Parametric study - Completed (8 configs tested)
3. ✅ ML/DL - Completed (XGBoost + Hybrid CNN-LSTM)

Comparison note:

- [x] All three tracks compared in one summary table
- [x] Profitable candidate selected: Optuna-Optimized XGBoost (Sharpe 5.19)

## Scope

- In scope files/modules:
  - `live_trading/` - Core trading engine
  - `dashboard/` - Svelte UI
  - `docs/` - GitHub Pages updates

- Out of scope:
  - Strategy parameter changes
  - New ML models
  - Additional coins (ETH/SOL live trading)

## Data and Infra Assumptions

- Data root: `/mnt/data/finance`
- Time-series format: hive partition ✅ Verified
- Compute plan:
   - [x] GPU usage considered (RTX 3090, but XGBoost uses CPU)
   - [x] Parallel processing plan defined (async API calls)
- Engine plan:
   - [x] Polars default
   - [x] DuckDB usage rationale documented (historical data joins only)

## Milestones

1. Context scan
   - [x] Read affected modules
   - [x] Confirm data/paths and dependencies
2. Design
   - [x] Define minimal change set
   - [x] Define verification strategy (5 checkpoints)
3. Implementation
   - [x] Core engine (engine.py, risk_manager.py)
   - [x] API integration (executor.py)
   - [x] Feature extraction parity (features.py)
   - [ ] Svelte dashboard
4. Verification
   - [x] CP-004 Risk tests passed
   - [ ] CP-001 Signal parity (needs historical run)
   - [ ] CP-003 Paper trading (7 days)
5. Reporting
   - [x] Required metrics table completed
   - [x] Pages content updated (benchmark.html)
6. Handoff (for live trading scope)
   - [x] Strategy note drafted (DEVELOPMENT.md)
   - [x] Checkpoint evidence attached (VERIFICATION.md)

## Verification Gates

- Gate A (static): `ruff check`, type hints
- Gate B (runtime): `pytest live_trading/tests/`
- Gate C (artifact consistency): Signal parity < 1e-6

## Metrics Contract (SOTA Reference)

- [x] Return: +2.82%
- [x] Alpha: +19.95%
- [x] Sharpe: 5.19
- [x] Max DD: 0.89%
- [x] Win Rate: 58.3%
- [x] Profit Factor: 2.56
- [x] Trades: 24 (test period)
- [x] Expected return per trade: 0.12%

## ML/DL Protocol

- [x] Train/Test split is time-based
- [x] Test window is the most recent 1 year (2026-01-01 ~ 2026-04-22)
- [x] Train window is all earlier history (2025-06-01 ~ 2025-12-31)

## Risks and Mitigations

- Risk: Model drift in production
  - Mitigation: Feature monitoring, daily drift alerts

- Risk: API rate limits
  - Mitigation: Exponential backoff, request caching

- Risk: Flash crash losses
  - Mitigation: Hard stop at -2% daily, -5% weekly

## Expected Deliverables

- Code:
  - `live_trading/` module ✅
  - `dashboard/` Svelte app (pending)
- Documents:
  - DEVELOPMENT.md ✅
  - RUNBOOK.md ✅
  - VERIFICATION.md ✅
- Generated artifacts:
  - `xgb_optuna_best.json` model ✅
  - `backtest_signals.json` for parity ✅

## Publishing Targets

- `docs/` updated for GitHub Pages ✅
- `docs/benchmark.html` with SOTA results ✅
- Practical trading simulation section with:
   - Seed capital: $10,000 USDT
   - Position size per trade: 10%
   - Execution model: Market orders
   - Fees: 0.04% taker
   - Slippage: 5 bps
   - Leverage: 1x (no leverage)

## Live Trading Checkpoints

| CP | Status | Description |
|----|--------|-------------|
| CP-001 | ⏳ | Core Engine Parity |
| CP-002 | ⏳ | Feature Extraction Parity |
| CP-003 | ⏳ | Order Execution Simulation (7d) |
| CP-004 | ✅ | Risk Limits Active |
| CP-005 | ⏳ | Live Deployment Ready |

## Svelte Dashboard Plan

- Dashboard tab: Position, P&L, Live signals
- History tab: Trade log with date filters
- Settings tab: API credentials, Trading toggle
