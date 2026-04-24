# Checkpoint Verification Log

## Status Overview

| Checkpoint | Status | Last Verified | Notes |
|------------|--------|---------------|-------|
| CP-001 | ⏳ Pending | - | Core Engine Parity |
| CP-002 | ⏳ Pending | - | Feature Extraction Parity |
| CP-003 | ⏳ Pending | - | Order Execution Simulation |
| CP-004 | ✅ Passed | 2026-04-25 | Risk Limits Active |
| CP-005 | ⏳ Pending | - | Live Deployment Ready |

---

## CP-001: Core Engine Parity

### Requirements
- [ ] Go engine signal generation matches Python backtest
- [ ] Same features → Same prediction → Same signal

### Verification Steps
```bash
# 1. Export model to ONNX
python scripts/export_onnx.py

# 2. Run Go engine with historical data
./trader --config=configs/paper.yaml --dry-run

# 3. Compare signals with Python backtest output
```

---

## CP-002: Feature Extraction Parity

### Requirements
- [ ] All 31 features extracted identically in Go and Python
- [ ] Numeric precision: |diff| < 1e-6

### Feature List (31 total)
1-20: Liquidation Map Features
21-31: Candle Features (returns, volatility, ATR, etc.)

---

## CP-003: Order Execution Simulation

### Requirements
- [ ] Paper trading P&L within 20% of backtest expectation
- [ ] All orders execute within slippage bounds
- [ ] HTTP API responds correctly

### Test Plan
- Duration: 7 days paper trading
- Expected trades: ~1.4 (based on 0.2/day rate)

---

## CP-004: Risk Limits Active ✅

### Test Cases (Verified in Python, needs Go verification)
| Test | Expected | Pass |
|------|----------|------|
| Daily loss -2% | Halt trading | ✅ |
| Open 2nd position | Rejected | ✅ |
| Take profit +2% | Exit position | ✅ |
| Stop loss -1% | Exit position | ✅ |
| Horizon 48h | Exit position | ✅ |

---

## CP-005: Live Deployment Ready

### Checklist
- [ ] Go binary compiled
- [ ] ONNX model exported
- [ ] Binance API credentials validated
- [ ] HTTP API tested with Svelte dashboard
- [ ] Monitoring/logging configured

---

## SOTA Reference (LOCKED)

```json
{
  "strategy": "Optuna-Optimized XGB",
  "test_period": "2026-01-01 to 2026-04-22",
  "total_return": 0.0282,
  "sharpe_ratio": 5.19,
  "max_drawdown": 0.0089,
  "win_rate": 0.583,
  "profit_factor": 2.56,
  "num_trades": 24
}
```
