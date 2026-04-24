# Operations Runbook

## Quick Start

### Paper Trading
```bash
cd live_trading

# Set environment variables
export BINANCE_API_KEY="your_testnet_key"
export BINANCE_API_SECRET="your_testnet_secret"
export BINANCE_TESTNET=true

# Run paper trading
python main.py --config configs/paper.yaml --capital 10000
```

### Live Trading
```bash
# WARNING: Uses real funds
export BINANCE_API_KEY="your_live_key"
export BINANCE_API_SECRET="your_live_secret"
export BINANCE_TESTNET=false

python main.py --config configs/production.yaml --capital 10000
```

---

## Pre-Flight Checklist

### Before First Run
- [ ] Model file exists at `models/xgb_optuna_best.json`
- [ ] API credentials set in environment
- [ ] Run parity tests: `python tests/test_parity.py`
- [ ] Run risk tests: `python tests/test_risk.py`
- [ ] Paper trade for 48+ hours with no errors

### Before Live Deployment
- [ ] All checkpoints verified (CP-001 through CP-005)
- [ ] Monitoring dashboard accessible
- [ ] Alert channels configured
- [ ] Emergency contacts documented
- [ ] Capital allocation approved

---

## Monitoring

### Key Metrics
| Metric | Expected | Alert Threshold |
|--------|----------|-----------------|
| Win Rate | 58% | < 45% over 20 trades |
| Sharpe Ratio | 5.19 | < 2.0 rolling 30d |
| Max Drawdown | 0.89% | > 3% |
| Trades/Day | 0.2 | > 2.0 or < 0.05 |

### Health Checks
```bash
# Check if process is running
pgrep -f "python main.py"

# Check recent logs
tail -100 logs/trading.log

# Check current position
curl http://localhost:8080/status  # If HTTP interface enabled
```

---

## Incident Response

### Trading Halted - Daily Loss Limit
1. Do NOT manually restart
2. Review trades that caused the loss
3. Check for anomalies (flash crash, API issues)
4. Wait for daily reset (UTC midnight) or manual override

```bash
# Manual override (use with caution)
# Edit state file to reset daily_pnl
```

### Trading Halted - Weekly Loss Limit
1. Full stop - requires manual intervention
2. Comprehensive review of all trades
3. Check for model drift
4. Re-validate with fresh backtest
5. Get approval before resuming

### API Errors
1. Check Binance status: https://www.binance.com/en/support/announcement
2. Verify API key is valid
3. Check rate limits
4. If maintenance, wait for completion

### Unexpected Position
1. Verify position matches internal state
2. If mismatch, close position on exchange
3. Sync internal state
4. Investigate cause before resuming

---

## Emergency Procedures

### Emergency Shutdown
```bash
# Graceful shutdown
kill -SIGTERM $(pgrep -f "python main.py")

# Force shutdown
kill -9 $(pgrep -f "python main.py")
```

### Emergency Position Close
```python
# Manual close via Python
import asyncio
from live_trading.src.executor import BinanceExecutor

async def emergency_close():
    executor = BinanceExecutor()
    result = await executor.close_position("BTCUSDT")
    print(result)
    await executor.close()

asyncio.run(emergency_close())
```

### Recovery After Crash
1. Check for open positions on exchange
2. Sync internal state with exchange
3. Review logs for cause
4. Fix issues before restart

---

## Maintenance

### Model Update
1. Train new model with updated data
2. Run full backtest
3. Verify results match or exceed SOTA
4. Update `models/xgb_optuna_best.json`
5. Run parity tests
6. Paper trade 48+ hours
7. Deploy to live

### Config Changes
1. Update config file
2. Restart trading engine
3. Monitor for 1 hour
4. Document changes

### Log Rotation
```bash
# Rotate logs weekly
mv logs/trading.log logs/trading.log.$(date +%Y%m%d)
gzip logs/trading.log.*

# Keep 30 days
find logs/ -name "*.gz" -mtime +30 -delete
```

---

## Contacts

| Role | Contact | When |
|------|---------|------|
| Developer | [TBD] | Code issues |
| Operations | [TBD] | System issues |
| Risk Manager | [TBD] | Loss limits hit |
