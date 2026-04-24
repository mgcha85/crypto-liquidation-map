# AGENTS.md - AI Agent Instructions

## Project Overview

**Crypto Liquidation Map** - ML-based crypto trading system using liquidation heatmap data.

**Tech Stack:**
- Python 3.12+ with type hints
- Polars (data processing - default)
- DuckDB (join-heavy/SQL pipelines only)
- XGBoost (primary ML model)
- PyTorch (CNN-LSTM hybrid)
- Plotly (visualization)
- aiohttp (async API calls)
- Svelte (live trading dashboard)

**Goal:** Production trading system based on liquidation map ML strategy.

## Repository Structure

```
crypto-liquidation-map/
├── src/liquidation_map/
│   ├── data/              # Data collection & processing
│   │   ├── downloader.py  # Binance Vision downloader
│   │   └── processor.py   # Raw → DuckDB pipeline
│   ├── analysis/          # Liquidation calculations
│   │   └── liquidation_map.py
│   ├── ml/                # ML pipeline
│   │   ├── features.py    # Feature extraction (31 features)
│   │   ├── labeling.py    # Triple barrier labeling
│   │   ├── backtest.py    # Backtesting engine
│   │   ├── multi_timeframe.py  # 5m/15m/1h ensemble
│   │   └── models/
│   │       ├── xgboost_model.py
│   │       └── hybrid_model.py  # CNN-LSTM-MLP
│   └── visualization/
│       └── heatmap.py
├── live_trading/          # Production trading module
│   ├── src/               # Core engine
│   ├── tests/             # Parity & risk tests
│   ├── configs/           # Paper/production configs
│   ├── checkpoints/       # Verification evidence
│   ├── models/            # Trained models
│   └── docs/              # Runbook & dev guide
├── dashboard/             # Svelte live trading UI
│   ├── src/routes/        # Dashboard, History, Settings
│   └── src/lib/           # API client, stores
├── scripts/               # CLI utilities & benchmarks
├── data/                  # Local storage (gitignored)
│   ├── raw/               # Downloaded ZIPs
│   ├── processed/         # Parquet files
│   ├── train/             # Models & results
│   └── cache/             # Request cache
├── docs/                  # GitHub Pages source
└── .opencode/             # AI workflow templates
```

## Data Sources

### Primary: /mnt/data/finance (Hive Partition)
- Path: `/mnt/data/finance/cryptocurrency/{exchange}/{symbol}/{timeframe}/`
- Formats: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- Use for: Training data, feature engineering

### Secondary: Binance Vision (Free API)
- Base URL: `https://data.binance.vision/`
- Open Interest: `data/futures/um/daily/openInterest/{SYMBOL}/`
- Liquidation: `data/futures/um/daily/liquidationSnapshot/{SYMBOL}/`
- Klines: `data/futures/um/daily/klines/{SYMBOL}/{INTERVAL}/`

### Live: Binance Futures API
- Testnet: `https://testnet.binancefuture.com`
- Production: `https://fapi.binance.com`

## SOTA Benchmark (LOCKED)

| Metric | Value | Notes |
|--------|-------|-------|
| Total Return | +2.82% | Test: 2026-01-01 ~ 2026-04-22 |
| Sharpe Ratio | 5.19 | Risk-adjusted |
| Max Drawdown | 0.89% | Risk limit reference |
| Win Rate | 58.3% | Consistency metric |
| Profit Factor | 2.56 | Win/Loss ratio |
| Alpha | +19.95% | vs -17.13% B&H |

### Optimal XGBoost Parameters (DO NOT MODIFY)
```python
{
    "max_depth": 8,
    "learning_rate": 0.0137,
    "n_estimators": 175,
    "subsample": 0.681,
    "colsample_bytree": 0.668,
    "gamma": 1.183,
    "reg_alpha": 2.367,
    "reg_lambda": 1.127,
}
```

### Triple Barrier (DO NOT MODIFY)
```python
{
    "profit_take": 0.02,  # 2%
    "stop_loss": 0.01,    # 1%
    "horizon": 48,        # hours
}
```

## Conventions

### Code Style
- Python 3.12+ with type hints
- Ruff for linting/formatting
- 100 char line length
- Polars over Pandas (always)
- Async for I/O operations
- GPU/parallel processing when feasible

### Naming
- snake_case for functions/variables
- PascalCase for classes
- SCREAMING_SNAKE_CASE for constants

### Testing
- pytest with pytest-asyncio
- Test files: `test_*.py`
- Parity tests required for live trading

### Commits
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Keep commits atomic

## ML/DL Protocol

- Train/Test split: TIME-BASED only
- Test window: Latest 1 year (most recent)
- Train window: All earlier history
- Feature count: 31 (20 liquidation + 11 candle)

## Strategy Track Order

1. Baseline algorithm backtest
2. Parametric study on key knobs
3. ML/DL enhancement
4. Cross-track comparison table
5. Profitable candidate → Hardening

## Live Trading Checkpoints

| CP | Description | Verification |
|----|-------------|--------------|
| CP-001 | Core Engine Parity | Signals match backtest |
| CP-002 | Feature Extraction Parity | |diff| < 1e-6 |
| CP-003 | Order Execution Simulation | 7-day paper trading |
| CP-004 | Risk Limits Active | Daily -2%, Weekly -5% |
| CP-005 | Live Deployment Ready | All systems go |

## Tool Permissions

**Allowed:**
- Read/edit: `src/`, `live_trading/`, `dashboard/`, `tests/`, `scripts/`, `docs/`
- Run: `pytest`, `ruff`, `mypy`
- Download from data.binance.vision
- Read from `/mnt/data/finance`

**Restricted (ask first):**
- Modifying `pyproject.toml` dependencies
- Creating new top-level directories
- Any network requests to paid APIs
- Modifying SOTA parameters

**Not allowed:**
- Modifying `.env.*` files with real credentials
- Push to main without verification
- Delete `data/` directory structure
- Suppress type errors (`as any`, `@ts-ignore`)

## Verification Gates

Before marking task complete:
- [ ] `ruff check src/ tests/` passes
- [ ] `pytest tests/` passes
- [ ] Parity tests pass (for live_trading changes)
- [ ] No TODO left in changed code
- [ ] Metrics table updated (for strategy changes)

## Publishing

- GitHub Pages: `docs/` directory
- Required pages: `index.html`, `benchmark.html`, `strategies.html`
- Comparison pages: `coins/{btc,eth,sol}.html`

## Svelte Dashboard Requirements

- Tabs: Dashboard, History, Settings
- Settings: API key, Secret key, Trading on/off toggle
- Dashboard: Position status, P&L, Signals
- History: Trade log with filters
