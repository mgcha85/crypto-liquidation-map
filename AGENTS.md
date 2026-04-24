# AGENTS.md - AI Agent Instructions

## Project Overview

**Crypto Liquidation Map** - Python tool for building liquidation heatmaps from Binance historical data.

**Tech Stack:**
- Python 3.11+
- Polars (data processing)
- DuckDB (local SQL database)
- Plotly (visualization)
- aiohttp (async downloads)

**Goal:** Build a Coinglass-like liquidation heatmap using free Binance Vision data.

## Repository Structure

```
crypto-liquidation-map/
├── src/liquidation_map/
│   ├── data/           # Data collection & processing
│   │   ├── downloader.py   # Binance Vision downloader
│   │   └── processor.py    # Raw → DuckDB pipeline
│   ├── analysis/       # Liquidation calculations
│   │   └── liquidation_map.py  # Core algorithm
│   └── visualization/  # Plotly heatmaps
│       └── heatmap.py
├── tests/              # pytest tests
├── scripts/            # CLI utilities
├── data/               # Local data storage (gitignored)
│   ├── raw/            # Downloaded ZIP files
│   ├── processed/      # Parquet files
│   └── cache/          # Request cache
└── docs/               # Documentation
```

## Conventions

### Code Style
- Python 3.11+ with type hints
- Ruff for linting/formatting
- 100 char line length
- Prefer Polars over Pandas
- Async for I/O operations

### Naming
- snake_case for functions/variables
- PascalCase for classes
- SCREAMING_SNAKE_CASE for constants

### Testing
- pytest with pytest-asyncio
- Test files: `test_*.py`
- Run: `pytest tests/`

### Commits
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Keep commits atomic

## Tool Permissions

**Allowed:**
- Read/edit files under `src/`, `tests/`, `scripts/`, `docs/`
- Run `pytest`, `ruff`, `mypy`
- Download from data.binance.vision

**Restricted (ask first):**
- Modifying `pyproject.toml` dependencies
- Creating new top-level directories
- Any network requests to paid APIs

**Not allowed:**
- Modifying `.env.*` files with real credentials
- Push to main without PR
- Delete `data/` directory structure

## Verification Gates

Before marking task complete:
- [ ] `ruff check src/ tests/` passes
- [ ] `mypy src/` passes
- [ ] `pytest tests/` passes
- [ ] No `TODO` left in changed code (or explicitly documented)

## Data Sources

### Binance Vision (Free)
Base URL: `https://data.binance.vision/`

**Paths:**
- Open Interest: `data/futures/um/daily/openInterest/{SYMBOL}/`
- Liquidation: `data/futures/um/daily/liquidationSnapshot/{SYMBOL}/`
- Klines: `data/futures/um/daily/klines/{SYMBOL}/{INTERVAL}/`

### Algorithm Reference

**Liquidation Price Formulas:**
```
Long:  Liq = Entry × (1 - 1/Leverage + MM)
Short: Liq = Entry × (1 + 1/Leverage - MM)
```

Where MM = Maintenance Margin rate (varies by leverage tier).
