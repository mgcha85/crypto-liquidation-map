---
name: btc-backtest-live-workflow
description: "Use when tasks involve BTC strategy research workflow, including baseline->parametric->ML/DL comparison, latest-1y test split, metrics reporting, pages publishing, and Go live-trading handoff with strategy notes/checkpoints. Trigger keywords: backtest, parametric, ML, DL, pages, hive partition, /mnt/data/finance, Go live trading, Svelte dashboard/history/settings."
---

# BTC Backtest to Live Workflow

## Purpose

Standardize how this repository executes research and hands off to live trading.

## Inputs

- Strategy hypothesis or change request
- Data under `/mnt/data/finance` in hive partition layout
- Target timeframe(s): `5m`, `15m`, `1h`, `4h`

## Workflow

1. Baseline algorithm backtest
2. Parametric study on key knobs
3. ML/DL enhancement with strict time split
   - Train: all history except latest 1 year
   - Test: latest 1 year
4. Compare three tracks in a single summary
5. If profitable candidate exists, run ideation for hardening:
   - Risk controls
   - Execution realism (fees/slippage/latency)
   - Robustness and regime sensitivity

## Performance and Engine Rules

- Use Polars by default.
- Use DuckDB only with explicit need (join-heavy/SQL-heavy pipeline or IO benefit).
- Use GPU and parallel processing aggressively where feasible.

## Required Metrics

- Return
- Alpha
- Sharpe
- Max DD
- Win Rate
- Profit Factor
- Trades
- Expected return per trade

## Documentation Outputs

- Publish baseline strategy and result pages in `pages/` for GitHub Pages.
- Keep `docs/` aligned if legacy pages still consume docs artifacts.
- For live-trading-bound work, completion requires:
  - Strategy note
  - Checkpoint list with validation evidence

## Live Trading Baseline (Svelte)

- Required tabs: Dashboard, History, Settings
- Settings minimum fields: exchange `api_key`, `secret_key`, trading on/off toggle
