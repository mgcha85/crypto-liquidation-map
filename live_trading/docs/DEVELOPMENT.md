# Live Trading Module - Development Guide

## Overview

Live trading implementation for the Crypto Liquidation Map strategy. This module translates the backtested ML strategy into production trading.

## SOTA Benchmark Reference (DO NOT DEVIATE)

### Winning Strategy: Optuna-Optimized XGBoost

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Return** | +2.82% | Test period: 2026-01-01 ~ 2026-04-22 |
| **Sharpe Ratio** | 5.19 | Risk-adjusted metric |
| **Max Drawdown** | 0.89% | Critical risk limit |
| **Win Rate** | 58.3% | Achieved consistency |
| **Profit Factor** | 2.56 | Wins/Losses ratio |
| **Num Trades** | 24 | ~0.2 trades/day |
| **Alpha vs B&H** | +19.95% | Beat -17.13% buy & hold |

### Optimal XGBoost Parameters (LOCKED)

```python
OPTIMAL_XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.013721713883241381,
    "n_estimators": 175,
    "subsample": 0.6814260926873211,
    "colsample_bytree": 0.6683699504391535,
    "gamma": 1.1830353419914514,
    "reg_alpha": 2.3666885979478645,
    "reg_lambda": 1.1269397326544215,
}
```

### Triple Barrier Parameters (LOCKED)

```python
BARRIER_CONFIG = {
    "profit_take": 0.02,  # 2% take profit
    "stop_loss": 0.01,    # 1% stop loss
    "horizon": 48,        # 48 hours max hold
}
```

### Position Sizing (LOCKED)

```python
POSITION_CONFIG = {
    "position_size_pct": 0.10,  # 10% of capital per trade
    "taker_fee_pct": 0.0004,    # 0.04% taker fee
    "slippage_bps": 5.0,        # 0.05% slippage assumption
}
```

---

## Development Checkpoints

### CP-001: Core Engine Parity
**Goal**: Ensure live engine produces IDENTICAL signals to backtest
**Verification**:
1. Run backtest on historical data → save signals to `checkpoints/backtest_signals.json`
2. Run live engine simulation on same data → save to `checkpoints/live_signals.json`
3. Compare: `assert backtest_signals == live_signals`

### CP-002: Feature Extraction Parity
**Goal**: Live feature extraction matches training pipeline exactly
**Verification**:
1. Extract features from historical window
2. Compare with cached training features
3. Max allowed difference: `|live - backtest| < 1e-6`

### CP-003: Order Execution Simulation
**Goal**: Paper trading produces expected results
**Verification**:
1. Run 7-day paper trading
2. Compare P&L with backtest expectation (within 20% tolerance)
3. All orders executed within slippage bounds

### CP-004: Risk Limits Active
**Goal**: Circuit breakers prevent catastrophic losses
**Verification**:
1. Daily loss limit: -2% → trading halts
2. Position limit: max 1 position at a time
3. Drawdown limit: -5% → full stop

### CP-005: Live Deployment Ready
**Goal**: Production system fully operational
**Verification**:
1. API credentials validated
2. Monitoring dashboard active
3. Alert system configured
4. Recovery procedures documented

---

## Architecture

```
live_trading/
├── src/
│   ├── __init__.py
│   ├── config.py           # LOCKED parameters from SOTA
│   ├── engine.py           # Core trading engine
│   ├── features.py         # Feature extraction (mirrors backtest)
│   ├── model.py            # XGBoost model wrapper
│   ├── executor.py         # Binance order execution
│   ├── risk_manager.py     # Position/risk limits
│   └── monitor.py          # Real-time monitoring
├── tests/
│   ├── test_parity.py      # CP-001, CP-002 verification
│   ├── test_execution.py   # CP-003 verification
│   └── test_risk.py        # CP-004 verification
├── configs/
│   ├── production.yaml     # Live trading config
│   └── paper.yaml          # Paper trading config
├── checkpoints/
│   ├── backtest_signals.json
│   └── live_signals.json
└── docs/
    ├── DEVELOPMENT.md      # This file
    ├── RUNBOOK.md          # Operations guide
    └── INCIDENT.md         # Incident response
```

---

## Development Rules

### Rule 1: NO PARAMETER CHANGES
The optimal parameters were found through rigorous backtesting. Do NOT modify:
- XGBoost hyperparameters
- Barrier config (PT/SL/Horizon)
- Position sizing

If you believe parameters need adjustment, create a new backtest first.

### Rule 2: PARITY FIRST
Every code change must pass parity tests before merge:
```bash
pytest tests/test_parity.py -v
```

### Rule 3: PAPER BEFORE LIVE
All changes must run in paper trading for 48+ hours before live deployment.

### Rule 4: LOGGING EVERYTHING
Every trade decision must be logged with:
- Timestamp
- Signal value
- Feature values
- Model confidence
- Order details

### Rule 5: FAIL SAFE
Default behavior on error = close positions and halt trading.

---

## Signal Generation Flow

```
1. Every 1 hour:
   ├── Fetch latest 50h of 1h candles
   ├── Fetch latest 50h of OI data
   ├── Extract 31 features (IDENTICAL to backtest)
   ├── Run XGBoost prediction
   ├── If signal != current_position:
   │   ├── Apply 5m→1h ensemble filter (optional)
   │   ├── Check risk limits
   │   └── Execute order
   └── Log everything

2. Continuous:
   ├── Monitor open position P&L
   ├── Check barrier conditions (TP/SL/Horizon)
   └── Execute exit if triggered
```

---

## API Requirements

### Binance Futures API
- HMAC SHA256 authentication
- Endpoints needed:
  - `GET /fapi/v1/klines` - Candle data
  - `GET /fapi/v1/openInterest` - OI snapshots
  - `POST /fapi/v1/order` - Place orders
  - `GET /fapi/v2/account` - Account info
  - `GET /fapi/v2/positionRisk` - Position info

### Rate Limits
- Weight limit: 2400/minute
- Order limit: 300/minute
- Klines: 5 weight per call
- OI: 5 weight per call

---

## Risk Management

### Position Limits
- Max 1 position at a time
- Position size: 10% of account
- Leverage: 1x (no leverage for safety)

### Loss Limits
- Per-trade: -1% (stop loss)
- Daily: -2% → halt trading
- Weekly: -5% → require manual restart

### Circuit Breakers
1. API error 3x → halt 5 minutes
2. Unexpected slippage > 0.5% → halt 1 hour
3. Model prediction error → halt until fix

---

## Monitoring

### Metrics to Track
- Realized P&L (per trade, daily, cumulative)
- Win rate (rolling 20 trades)
- Sharpe ratio (rolling 30 days)
- Max drawdown (all time)
- Trade latency (order to fill)
- Feature drift (compare to training distribution)

### Alerts
- Trade executed → Slack notification
- Daily P&L summary → Email
- Loss limit hit → Urgent alert
- System error → PagerDuty

---

## Recovery Procedures

### Scenario: Unexpected Shutdown
1. On restart, check for open positions
2. Sync state with exchange
3. Resume normal operation

### Scenario: Model Drift Detected
1. Halt new trades
2. Compare live features with training distribution
3. If drift > threshold, retrain model
4. Validate with parity tests
5. Resume trading

### Scenario: Exchange Maintenance
1. Close positions before maintenance window
2. Halt trading during maintenance
3. Resume after API health check
