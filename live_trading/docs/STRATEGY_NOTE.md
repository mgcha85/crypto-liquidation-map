# Strategy Note: XGBoost Liquidation Map Strategy

## Overview

ML-enhanced trading strategy using liquidation map features for BTC/USDT futures.

## Model Architecture

- **Algorithm**: XGBoost binary classification
- **Features**: 31 (20 liquidation + 11 candle)
- **Target**: Triple barrier labels (PT=2%, SL=1%, Horizon=48h)
- **Training**: Optuna hyperparameter optimization (100 trials)

## SOTA Parameters (LOCKED)

```yaml
max_depth: 8
learning_rate: 0.0137
n_estimators: 175
subsample: 0.8
colsample_bytree: 0.8
gamma: 0.1
reg_alpha: 0.1
reg_lambda: 1.0
```

## Backtest Results

| Metric | Value |
|--------|-------|
| Total Return | +2.82% |
| Sharpe Ratio | 5.19 |
| Max Drawdown | 0.89% |
| Win Rate | 58.3% |
| Profit Factor | 2.56 |
| Total Trades | 12 |

**Test Period**: 2024-01-01 ~ 2024-12-31 (1 year)
**Train Period**: 2020-01-01 ~ 2023-12-31 (4 years)

## Feature Groups

### Liquidation Features (20)
- total_intensity, long_intensity, short_intensity
- long_short_ratio, above_below_ratio
- near_1pct/2pct/5pct_concentration
- cluster distances and volumes
- entropy, skewness

### Candle Features (11)
- return_1h/6h/12h/24h
- volatility_6h/24h
- atr_24h, volume_ma_ratio
- wick_ratio_upper/lower, price_position

## Risk Management

- **Position Size**: 10% of equity
- **Daily Loss Limit**: 2%
- **Weekly Loss Limit**: 5%
- **Max Positions**: 1

## Execution

- **Interval**: 1 hour
- **Order Type**: Market orders
- **Exit Logic**: Triple barrier (PT/SL/Horizon)

## Deployment

```bash
cd live_trading
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
./trader --config=configs/paper.yaml --capital=10000 --static=../dashboard/build
```

## Validation Checkpoints

- [x] CP-001: Model training complete
- [x] CP-002: ONNX export verified
- [x] CP-003: Go inference tested
- [x] CP-004: Paper trading API working
- [ ] CP-005: 1-week paper trading validation
- [ ] CP-006: Production deployment

## Notes

- OI data may return API errors (202) - handled gracefully
- Telegram notifications on trade close
- Dashboard at http://localhost:8080
