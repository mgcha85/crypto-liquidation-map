# IMPLEMENT.md

## Task Summary

- Task: Go Live Trading Engine + Svelte Dashboard Integration
- Date: 2026-04-25
- Agent: Sisyphus

## Data and Runtime

- Data root used: `/mnt/data/finance`
- Hive partition verified: Yes (BTCUSDT 15m data)
- Compute used (GPU/parallel): GPU for XGBoost training
- Engine choice: Go for live trading, Python for backtest/ML

## Changes Applied

### Go Trading Engine (`live_trading/`)

- File: `cmd/trader/main.go`
  - Entry point with graceful shutdown, config loading, static file serving

- File: `internal/config/config.go`
  - SOTA parameters LOCKED (Optuna Trial #24)
  - max_depth=8, learning_rate=0.0137, n_estimators=175
  - Triple barrier: PT=2%, SL=1%, Horizon=48h

- File: `internal/engine/engine.go`
  - Core trading loop with 15m tick interval
  - Feature extraction (31 features: 20 liquidation + 11 candle)
  - Signal generation via ONNX inference

- File: `internal/binance/client.go`
  - Binance Futures API (testnet/production)
  - Kline, OI, funding rate, ticker fetching
  - Order placement (market orders)

- File: `internal/model/predictor.go`
  - ONNX Runtime inference
  - XGBoost binary classification (long vs short)
  - Input: 31 float32 features, Output: label + probabilities

- File: `internal/risk/manager.go`
  - Position management (entry/exit)
  - PnL tracking, trade history
  - Daily loss limit, max position size

- File: `internal/api/server.go`
  - HTTP API: /api/status, /api/trades, /api/metrics, /api/start, /api/stop
  - SPA handler for Svelte dashboard

### Svelte Dashboard (`dashboard/`)

- File: `src/routes/+page.svelte`
  - Dashboard: real-time status, position info, equity

- File: `src/routes/history/+page.svelte`
  - Trade history table with PnL

- File: `src/routes/settings/+page.svelte`
  - API key/secret input, testnet toggle
  - Telegram notification toggle (pending backend)

- File: `src/lib/api.ts`
  - API client using relative /api path

- File: `src/lib/stores.ts`
  - Svelte writable stores for state management

### ONNX Export

- File: `live_trading/scripts/export_onnx.py`
  - XGBoost to ONNX conversion
  - 31 input features, binary classification output

- File: `live_trading/models/xgb_optuna_best.onnx`
  - Exported model (239KB)

## SOTA Benchmark (LOCKED)

| Metric | Value |
|--------|-------|
| Total Return | +2.82% |
| Sharpe Ratio | 5.19 |
| Max Drawdown | 0.89% |
| Win Rate | 58.3% |
| Profit Factor | 2.56 |
| Total Trades | 12 |

## Verification

- Commands run:
  - `go build ./cmd/trader` → Success
  - `npm run build` (dashboard) → Success
  - `./trader --config=configs/paper.yaml --capital=10000` → Success
  - ONNX inference test → confidence=0.60, signal=-1 (SHORT)

- HTTP API tested:
  - `/api/status` → position info, equity
  - `/api/trades` → trade history
  - `/api/metrics` → PnL metrics

- Dashboard integration:
  - Static files served from Go server
  - SPA routing working

## FE-BE Serving Integration

- F/E build command: `cd dashboard && npm run build`
- Build artifact path: `dashboard/build/`
- B/E serving: `./trader --static=../dashboard/build`
- Serving verification: ✅ HTML served at /, API at /api/*

## Pending Items

1. Telegram notification integration
2. Settings API endpoint (/api/settings)
3. Testnet paper trading validation (1 week)
4. Production deployment

## Live Trading Readiness

- Strategy note: `live_trading/docs/STRATEGY_NOTE.md` (pending)
- Checkpoint evidence: `live_trading/checkpoints/VERIFICATION.md`
- ONNX model: `live_trading/models/xgb_optuna_best.onnx` ✅
- Go binary: `live_trading/trader` ✅
- Dashboard build: `dashboard/build/` ✅

## Next Steps

1. Implement Telegram notification in Go engine
2. Add /api/settings endpoint for runtime config
3. Create strategy note document
4. Run paper trading for validation
