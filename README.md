# Crypto Liquidation Map

Build liquidation heatmaps like Coinglass using free Binance Vision data.

## Overview

This tool creates liquidation maps by:
1. Downloading historical Open Interest and price data from Binance Vision
2. Estimating position entries based on OI changes
3. Calculating liquidation prices for various leverage levels
4. Visualizing as interactive heatmaps

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env.dev

# Run tests
pytest tests/
```

## Project Structure

```
src/liquidation_map/
├── data/           # Download & process Binance data
├── analysis/       # Liquidation map calculations
└── visualization/  # Plotly heatmaps
```

## Data Sources

Uses free data from [Binance Vision](https://data.binance.vision/):
- `data/futures/um/daily/openInterest/` - Open Interest snapshots
- `data/futures/um/daily/liquidationSnapshot/` - Liquidation events
- `data/futures/um/daily/klines/` - OHLCV price data

## Tech Stack

- **Polars** - Fast DataFrame operations
- **DuckDB** - Local SQL analytics
- **Plotly** - Interactive visualizations
- **aiohttp** - Async data downloads

## License

MIT
