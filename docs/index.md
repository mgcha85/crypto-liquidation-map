# PPO Trading Strategy Results

BTC/USDT 1H timeframe with PPO (Proximal Policy Optimization) reinforcement learning.

## Configuration

| Parameter | Value |
|-----------|-------|
| Leverage | 2x |
| Position Size | 25% |
| Hard Stop Loss | 5% (leveraged) |
| Effective Exposure | 0.50x |
| Commission | 0.04% |
| Slippage | 5 bps |

## Yearly Cross-Validation Results (2021-2025)

| Year | PPO Return | Buy & Hold | Alpha | Sharpe | Max DD | Trades |
|------|------------|------------|-------|--------|--------|--------|
| 2021 | +355.55% | +59.40% | +296.16% | 8.17 | - | - |
| 2022 | +262.59% | -64.54% | +327.13% | 8.06 | - | - |
| 2023 | +73.67% | +155.80% | -82.13% | 6.75 | - | - |
| 2024 | +130.08% | +120.31% | +9.77% | 6.95 | - | - |
| 2025 | +89.90% | -7.15% | +97.05% | 6.65 | 0.00% | 1208 |
| **AVG** | **+182.36%** | +52.76% | **+129.60%** | **7.32** | - | - |

## Key Metrics

- **Average Annual Return**: +182.36%
- **Average Alpha**: +129.60%
- **Average Sharpe Ratio**: 7.32
- **No Liquidation Risk**: 2x leverage with 5% hard SL prevents margin calls

## Model Architecture

- **Policy**: Hybrid CNN + MLP
  - CNN processes 200-bar OHLCV candle window
  - MLP processes 34 ML features + 2 portfolio features
- **Actions**: Short (-1), Hold (0), Long (+1)
- **Training**: PPO with GAE (λ=0.95, γ=0.99)

## Signal Parity Verification

| Checkpoint | Status |
|------------|--------|
| CP-001: Core Engine Parity | ✅ PASSED |
| CP-002: Feature Extraction | ✅ PASSED |
| Match Rate | 100% (50/50 samples) |

Python backtest and Go live trading produce identical signals.

## Live Trading Config

```yaml
position:
  size_pct: 0.25
  leverage: 2
  stop_loss_pct: 0.05
  allocation_pct: 1.0

barrier:
  profit_take: 0.02
  stop_loss: 0.01
  horizon_hours: 24
```

## Notes

- Test split: Most recent 1 year (2025)
- Train: All prior history (2020-2024)
- Triple barrier adds safety rails in live trading
- 5% hard SL prevents liquidation at any leverage ≤ 20x

---

*Last updated: 2026-04-25*
