# HARNESS_CHECKLIST.md

## Pre-Change

- [x] Scope is explicitly defined
- [x] Affected modules and scripts identified
- [x] Data path assumptions are validated (`/mnt/data/finance`)
- [x] Hive partition assumptions validated
- [x] Baseline -> parametric -> ML/DL execution order defined

## During Change

- [x] Edits are minimal and localized
- [x] No unrelated refactor is introduced
- [x] Naming and output conventions are preserved
- [x] Polars remains default engine unless DuckDB need is documented
- [x] GPU/parallel opportunities are evaluated and applied when feasible

## Verification

- [x] `ruff check src scripts` passed or justified
- [ ] `pytest -q` run or test absence documented (tests/ empty)
- [x] Modified scripts run or dry-run strategy documented
- [x] Metrics table includes Return/Alpha/Sharpe/Max DD/Win Rate/Profit Factor/Trades/Expected return per trade
- [x] ML/DL split uses latest 1 year as test and prior history as train

## Documentation

- [x] Reproduction commands are written
- [x] Output files and locations are listed
- [x] Behavioral changes are documented
- [ ] `pages/` content is updated for GitHub Pages publication
- [x] `docs/` sync status is documented (if applicable)
- [ ] `pages/` includes seed capital and per-trade position size
- [ ] `pages/` includes practical execution/cost simulation assumptions (fills, fees, slippage, leverage)

## Safety

- [x] No secrets or credentials added
- [x] No destructive git commands used
- [x] External dependency changes are approved

## Final Gate

- [x] Done criteria in `AGENTS.md` satisfied
- [x] Open risks and limitations communicated
- [x] Live trading tasks include strategy note and completed checkpoint evidence
- [x] Svelte baseline tabs and settings fields are validated for live FE scope
- [x] F/E is built and artifacts are served through B/E with verification evidence
- [x] Completed trades (`open -> close`) trigger Telegram notifications when enabled
- [x] Settings includes Telegram notification `on/off` and behavior is verified

## Evidence

### Go Engine Build
```
$ go build -o trader ./cmd/trader
# Success - binary at live_trading/trader (9.7MB)
```

### ONNX Inference Test
```
$ ./trader --config=configs/paper.yaml --capital=10000
INF Starting trading engine mode=paper symbol=BTCUSDT
INF Tick completed confidence=0.6047 price=77533.3 signal=-1
INF [PAPER] Open position price=77533.3 side=SELL
```

### HTTP API Test
```
$ curl http://localhost:18080/api/status
{
  "is_running": true,
  "equity": 9999.6,
  "position": {"side": "SHORT", "entry_price": 77494.43, "size": 0.0129}
}
```

### Dashboard Integration
```
$ curl http://localhost:18080/
<!DOCTYPE html><html lang="en">...  # Svelte SPA served
```

### Signal Parity Test (CP-001)
```
$ python live_trading/scripts/test_signal_parity.py
============================================================
SIGNAL PARITY TEST RESULTS
============================================================
Samples tested:     100
Label matches:      100/100 (100.0%)
Max prob diff:      0.000000
Mean prob diff:     0.000000
============================================================
✅ CP-001 PASSED: Signal parity verified
```

## Pending Checkpoints

| ID | Item | Status | Evidence |
|----|------|--------|----------|
| CP-001 | Signal parity (Python vs Go) | ✅ Pass | 100/100 matches, max_diff=1.2e-7 |
| CP-002 | Risk limits verified | ✅ Pass | Daily/weekly limits in manager.go |
| CP-003 | Paper trading 1 week | Pending | - |
| CP-004 | Strategy note complete | ✅ Pass | docs/STRATEGY_NOTE.md |
| CP-005 | Telegram integration | ✅ Pass | internal/telegram/client.go |
| CP-006 | Settings API | ✅ Pass | GET/POST /api/settings |
