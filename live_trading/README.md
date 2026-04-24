# Live Trading (Go)

Go-based live trading engine for the Crypto Liquidation Map strategy.

## Build

```bash
cd live_trading
go mod tidy
go build -o trader ./cmd/trader
```

## Run

```bash
# Paper trading
export BINANCE_API_KEY="your_testnet_key"
export BINANCE_API_SECRET="your_testnet_secret"
./trader --config=configs/paper.yaml --capital=10000

# Live trading (CAUTION: real funds)
./trader --config=configs/production.yaml --capital=10000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Current engine state |
| `/trades` | GET | Trade history |
| `/metrics` | GET | Performance metrics |
| `/start` | POST | Start engine |
| `/stop` | POST | Stop engine |

## ONNX Model

Export XGBoost model to ONNX:
```bash
python scripts/export_onnx.py
```

## Project Structure

```
live_trading/
├── cmd/trader/main.go       # Entry point
├── internal/
│   ├── config/config.go     # SOTA parameters (LOCKED)
│   ├── engine/engine.go     # Core trading loop
│   ├── binance/client.go    # Binance Futures API
│   ├── model/predictor.go   # ONNX inference
│   ├── risk/manager.go      # Position/risk management
│   └── api/server.go        # HTTP API for dashboard
├── configs/
│   ├── paper.yaml
│   └── production.yaml
├── models/                   # ONNX models
├── docs/
└── checkpoints/
```

## SOTA Reference (LOCKED)

| Metric | Value |
|--------|-------|
| Total Return | +2.82% |
| Sharpe Ratio | 5.19 |
| Max Drawdown | 0.89% |
| Win Rate | 58.3% |
| Profit Factor | 2.56 |
