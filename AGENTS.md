# AGENTS.md

This repository is configured for OpenCode-style agent workflows.

## 1) Mission and Scope

- Primary goal: maintain a reproducible BTC strategy research stack and production handoff path.
- Implementation split:
  - Backtest and research: Python
  - Live trading engine: Go
  - Live trading frontend: Svelte
- Core outputs:
  - Reproducible research outputs and strategy comparisons
  - Strategy notes and validation checkpoints for live deployment
  - Safe, reviewable code changes in `src/crypto_backtest/`, `scripts/`, and future Go runtime modules

## 2) Project Map

- Core package: `src/liquidation_map/`
  - `analysis/liquidation_map.py`: Liquidation map calculations
  - `visualization/heatmap.py`: Interactive Plotly heatmaps
  - `data/`: Download and process Binance data
- Backtest package: `src/crypto_backtest/`
  - `long_strategy.py`: L-shape entry/exit logic and backtest engine
  - `long_parametric.py`: parameter grid study and ranking
  - `data_loader.py`: minute -> timeframe resampling and indicator loading
  - `features.py`, `models_tree.py`, `models_deep.py`: ML/DL pipeline
  - `report.py`, `storage.py`: reporting and persistence
- Executable scripts: `scripts/`
  - 5m/15m/4h/1h analyses and multi-asset studies
- Live trading system: `live_trading/`
  - `cmd/trader/main.go`: entry point with graceful shutdown
  - `internal/config/config.go`: SOTA parameters (LOCKED)
  - `internal/engine/engine.go`: core trading loop with feature extraction
  - `internal/binance/client.go`: Binance Futures API (testnet/production)
  - `internal/model/predictor.go`: ONNX model inference (XGBoost)
  - `internal/risk/manager.go`: position/risk management
  - `internal/api/server.go`: HTTP API server + static file serving
  - `configs/paper.yaml`, `configs/production.yaml`: runtime configs
  - `models/xgb_optuna_best.onnx`: trained model for inference
  - `scripts/export_onnx.py`: XGBoost to ONNX conversion
- Frontend: `dashboard/`
  - `src/routes/+page.svelte`: Dashboard (real-time status, position)
  - `src/routes/history/+page.svelte`: Trade history
  - `src/routes/settings/+page.svelte`: API keys, Telegram toggle
  - `src/lib/api.ts`: API client for engine communication
  - `src/lib/stores.ts`: Svelte state management
  - `build/`: Static SPA build (served by Go engine)
- Published artifacts:
  - `pages/`: canonical GitHub Pages source for strategy/result pages
  - `docs/`: legacy or supporting artifacts (keep in sync when both are used)

## 3) Data Contract

- Root data path: `/mnt/data/finance`
- Time-series layout: hive partition
- Agent behavior:
  - Never hardcode alternate data roots without explicit approval
  - Validate partition/timeframe availability before long runs
  - Keep loaders deterministic and explicit about partition columns

## 4) Agent Modes

Use two-step workflow by default.

1. Plan mode (read-only)
   - Understand impacted modules and data dependencies
   - Draft or update `.opencode/PLAN.md`
2. Build mode (write)
   - Apply minimal code/doc changes
   - Record decisions and deltas in `.opencode/IMPLEMENT.md`

If the user explicitly asks for immediate implementation, skip directly to Build mode.

## 5) Context Loading Order

Before editing code, read context in this order:

1. `README.md`
2. `pyproject.toml`
3. Target module in `src/crypto_backtest/`
4. Related script in `scripts/`
5. Matching report/result docs in `pages/` and `docs/`

## 6) Runtime and Performance

- Backtest data/compute defaults:
  - Use `polars` as the primary dataframe/query engine
  - Use `duckdb` only when query ergonomics or IO pattern clearly benefit
- Performance policy:
  - Actively use GPU and parallel processing where feasible
  - Prefer vectorized operations and batched model training
  - Avoid single-thread bottlenecks in large parametric or ML/DL runs

## 7) Research Workflow

Run strategy tracks in strict sequence, then compare:

1. Baseline algorithm backtest
2. Parametric study
3. ML/DL enhancement

After sequence completion:

- Compare all three tracks side-by-side
- If any track is profitable, select candidate strategy and run an ideation phase:
  - Identify robustness improvements
  - Identify risk-control improvements
  - Identify execution and slippage realism improvements

## 8) Metrics Standard

Backtest result tables should include at least:

- Return
- Alpha
- Sharpe
- Max DD
- Win Rate
- Profit Factor
- Trades
- Expected return per trade

## 9) ML/DL Split Protocol

- Time split is fixed:
  - Test: most recent 1 year
  - Train: all prior history
- Do not use random shuffle split as the primary report split.

## 10) Publishing Policy

- Baseline strategy description and core results must be organized in `pages/`.
- GitHub Pages publishing content must be reproducible from repository artifacts.
- If `docs/` is still used by legacy pages, keep headline metrics and conclusions aligned.
- Every published strategy/result page must include practical trading simulation assumptions:
  - Seed capital (initial capital)
  - Position size per trade (allocation ratio or fixed amount)
  - Execution model summary (entry/exit fill assumptions)
  - Trading cost assumptions (fees/slippage) and any leverage assumptions

## 11) Live Trading Readiness Gate

Before declaring implementation complete for Go live trading:

1. Backtest strategy note is written and versioned
2. Checkpoint list is completed with evidence
3. All required validations are passed or explicitly waived with rationale

## 12) Live Frontend Baseline

Svelte frontend must include baseline tabs:

- Dashboard
- History
- Settings

Settings must include at least:

- Exchange API key
- Exchange secret key
- Trading on/off toggle
- Telegram notification on/off toggle

Live execution notification rule:

- When a trade is completed (`open -> close`), the system must send a Telegram message if Telegram notifications are enabled.

Deployment rule:

- F/E must be built and the build artifacts served through B/E.

## 13) Commands and Verification

Environment setup:

```bash
source .venv/bin/activate
pip install -e .
```

Quality gates (run when relevant):

```bash
ruff check src scripts
python -m pytest -q
```

Notes:

- `tests/` is currently empty. If logic is changed, add focused tests first.
- Many scripts use hard-coded external data paths. Validate and migrate toward `/mnt/data/finance` contract.

## 14) Safe Change Policy

- Keep changes minimal and local to the requested scope.
- Do not refactor unrelated files.
- Preserve existing output file names unless explicit migration is requested.
- Prefer deterministic outputs and explicit config over hidden defaults.

## 15) Data and Experiment Conventions

- Timeframes in this repo: `5m`, `15m`, `1h`, `4h`
- Typical strategy knobs:
  - `breakout_ma`
  - `consolidation_bars`
  - `consolidation_range_pct`
  - `drop_threshold_pct`
  - `take_profit_pct`
  - `stop_loss_pct`
  - `half_close_enabled`, `half_close_pct`
- Store generated tables and json with clear timeframe prefix under publish artifacts.

## 16) Tooling and Permissions

- Allowed by default:
  - File read/write inside repository
  - Python execution in local venv
  - Linting and tests
- Ask before:
  - Installing new dependencies
  - Large-scale rewrites
  - Deleting or renaming result artifacts in `pages/` or `docs/`
  - Any destructive git action

## 17) Definition of Done

A task is done when all are true:

1. Requested code/docs are implemented
2. Relevant checks pass (or failures are explained)
3. Baseline -> parametric -> ML/DL comparison is documented (when in scope)
4. Required metrics are reported
5. Output paths and reproduction steps are documented
6. Strategy note and checkpoints are complete for live-trading-bound tasks
7. For live scope, F/E build artifacts are integrated into B/E serving path
8. For live scope, completed trades (`open -> close`) trigger Telegram notifications when enabled
9. `.opencode/IMPLEMENT.md` is updated for non-trivial tasks

## 18) Output Style for Agents

- Summaries should include:
  - Changed files
  - Behavioral impact
  - How to run or verify
- For reviews, list findings first by severity with exact file references.
