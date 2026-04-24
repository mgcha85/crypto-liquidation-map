# Strategy Note - Optuna-Optimized XGBoost

## Strategy Summary

**Name:** Liquidation Map XGBoost Strategy  
**Version:** 1.0.0  
**Status:** Ready for Paper Trading  
**Last Updated:** 2026-04-25

## Core Concept

Uses liquidation heatmap features combined with price action to predict short-term BTC price direction. Positions are managed with triple barrier method (take profit, stop loss, time horizon).

## Signal Logic

```
IF model_prediction == 1 (BUY):
    → Open LONG position
IF model_prediction == -1 (SELL):
    → Open SHORT position
IF model_prediction == 0 (HOLD):
    → Close any open position
```

## Position Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Take Profit | 2% | Optimal from parametric study |
| Stop Loss | 1% | 2:1 reward-risk ratio |
| Max Hold | 48 hours | Avoid overnight risk accumulation |
| Position Size | 10% of capital | Conservative sizing |
| Leverage | 1x | No leverage for risk control |

## Feature Inputs (31 total)

### Liquidation Map Features (20)
- Total/Long/Short intensity
- Long/Short ratio
- Above/Below ratio
- Concentration at 1%/2%/5% from price
- Largest cluster distances and volumes
- Top 3 cluster distances (both sides)
- Entropy and skewness

### Price Action Features (11)
- Returns: 1h, 6h, 12h, 24h
- Volatility: 6h, 24h
- ATR 24h
- Volume MA ratio
- Wick ratios (upper/lower)
- Price position in 24h range

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost Classifier |
| Objective | binary:logistic |
| max_depth | 8 |
| learning_rate | 0.0137 |
| n_estimators | 175 |
| subsample | 0.681 |
| colsample_bytree | 0.668 |
| gamma | 1.183 |
| reg_alpha | 2.367 |
| reg_lambda | 1.127 |

## Expected Performance

| Metric | Backtest | Target (Live) |
|--------|----------|---------------|
| Annual Return | ~8.5%* | >5% |
| Sharpe Ratio | 5.19 | >3.0 |
| Max Drawdown | 0.89% | <3% |
| Win Rate | 58.3% | >50% |
| Profit Factor | 2.56 | >1.5 |

*Annualized from 111-day test period

## Risk Controls

### Hard Limits (Circuit Breakers)
- Daily loss > 2% → Trading halted until next day
- Weekly loss > 5% → Trading halted, manual review required
- API errors 3x consecutive → Pause 5 minutes

### Soft Limits (Warnings)
- Win rate drops below 45% (rolling 20 trades)
- Sharpe drops below 2.0 (rolling 30 days)
- Feature drift detected (KS test p < 0.05)

## Execution Parameters

| Parameter | Value |
|-----------|-------|
| Order Type | Market |
| Taker Fee | 0.04% |
| Slippage Assumption | 5 bps |
| Update Frequency | 1 hour |
| Data Lookback | 50 hours |

## Deployment Checklist

- [x] Model trained and validated
- [x] Risk manager implemented
- [x] CP-004 (Risk Limits) verified
- [ ] CP-001 (Signal Parity) pending
- [ ] CP-003 (Paper Trading) pending
- [ ] Svelte dashboard implemented
- [ ] Alert system configured

## Known Limitations

1. **Single Symbol**: Currently BTCUSDT only
2. **Hourly Granularity**: May miss intra-hour opportunities
3. **Market Orders**: No limit order optimization
4. **No Regime Detection**: Same parameters across all market conditions

## Future Improvements

1. Multi-symbol support (ETH, SOL)
2. Adaptive position sizing based on confidence
3. Regime-aware parameter switching
4. Ensemble with 5m/15m timeframes
