# Liquidation Map Backtest Report

## Overview

This report documents the backtesting methodology and results for the liquidation map trading strategy.

**Data Period**: 2020-09-01 ~ 2026-04-24 (5.7 years)  
**Train Period**: 2020-09-03 ~ 2025-12-31 (7,784 samples)  
**Test Period**: 2026-01-01 ~ 2026-04-22 (448 samples)

---

## 1. Backtest Methodology

### 1.1 Data Pipeline

```
Binance Vision → ZIP → Parquet (Hive Partitioned) → Features → Labels → Train/Test Split
```

**Data Sources:**
- Open Interest (OI): 5-minute intervals (668,189 rows)
- Klines (OHLCV): 1-hour intervals (49,464 rows)
- Period: 2020-09-01 ~ 2026-04-24 (5.7 years)

### 1.2 Feature Engineering

**50-hour lookback window** generates 31 features:

| Category | Features | Description |
|----------|----------|-------------|
| Liquidation Map (20) | `total_intensity`, `long_intensity`, `short_intensity` | Total liquidation volume |
| | `long_short_ratio`, `above_below_ratio` | Ratio metrics |
| | `near_1/2/5pct_concentration` | Volume near current price |
| | `largest_long/short_cluster_distance/volume` | Largest cluster info |
| | `top3_long/short_dist_1/2/3` | Top 3 clusters |
| | `entropy`, `skewness` | Distribution shape |
| Candle (11) | `return_1h/6h/12h/24h` | Returns |
| | `volatility_6h/24h`, `atr_24h` | Volatility |
| | `volume_ma_ratio` | Volume ratio |
| | `wick_ratio_upper/lower`, `price_position` | Candle patterns |

### 1.3 Labeling (Triple-Barrier Method)

```
Profit Take (PT): +2%
Stop Loss (SL): -1%
Horizon: 24 hours
```

| Label | Condition |
|-------|-----------|
| +1 (Buy) | Price hits PT first within horizon |
| -1 (Sell) | Price hits SL first within horizon |
| 0 (Hold) | Neither hit, use return sign |

### 1.4 Train/Test Split

```
|-------- TRAIN (5.3 years) --------|-------- TEST (4 months) --------|
     2020-09 ~ 2025-12                     2026-01 ~ 2026-04
```

- **Temporal split**: No data leakage between train and test
- **Sample rate**: Every 6 hours (reduces computation, maintains signal)

### 1.5 Backtest Execution

| Parameter | Value |
|-----------|-------|
| Initial Capital | $100,000 |
| Position Size | 10% of capital |
| Taker Fee | 0.04% |
| Slippage | 5 bps |
| Max Positions | 1 |

---

## 2. Model Performance

### 2.1 XGBoost (5-Year Training)

**Hyperparameters:**
- `max_depth`: 4
- `learning_rate`: 0.05
- `n_estimators`: 150
- `scale_pos_weight`: 1.61 (class imbalance correction)

**Classification Metrics:**

| Metric | Value |
|--------|-------|
| Accuracy | 58.5% |
| Buy Precision | 37.9% |
| Buy Recall | 34.9% |
| Buy F1 | 36.3% |

### 2.2 Top 15 Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `largest_long_cluster_volume` | 0.0444 |
| 2 | `near_5pct_concentration` | 0.0381 |
| 3 | `short_intensity` | 0.0380 |
| 4 | `skewness` | 0.0374 |
| 5 | `atr_24h` | 0.0373 |
| 6 | `total_intensity` | 0.0372 |
| 7 | `top3_short_dist_1` | 0.0364 |
| 8 | `volatility_24h` | 0.0355 |
| 9 | `top3_long_dist_1` | 0.0353 |
| 10 | `above_below_ratio` | 0.0341 |
| 11 | `volume_ma_ratio` | 0.0340 |
| 12 | `volatility_6h` | 0.0337 |
| 13 | `return_12h` | 0.0335 |
| 14 | `near_1pct_concentration` | 0.0332 |
| 15 | `long_intensity` | 0.0330 |

**Key Insight**: Liquidation cluster volume and concentration are the most predictive features.

---

## 3. Backtest Results (2026 Test Period)

### 3.1 Strategy Comparison

| Strategy | Accuracy | Return | Sharpe | Max DD | Win Rate | Trades | Alpha |
|----------|----------|--------|--------|--------|----------|--------|-------|
| **XGBoost** | 58.5% | **+1.30%** | 1.88 | 1.52% | 55.0% | 100 | **+12.09%** |
| **CNN** | 47.1% | -0.24% | -0.32 | 2.32% | 55.8% | 95 | +10.56% |
| Buy & Hold | N/A | -10.80% | N/A | N/A | N/A | 1 | 0% |

### 3.2 Performance Metrics

| Metric | XGBoost | CNN |
|--------|---------|-----|
| **Total Return** | +1.30% | -0.24% |
| **Sharpe Ratio** | 1.88 | -0.32 |
| **Max Drawdown** | 1.52% | 2.32% |
| **Win Rate** | 55.0% | 55.8% |
| **Number of Trades** | 100 | 95 |
| **Alpha vs B&H** | **+12.09%** | +10.56% |

### 3.3 Analysis

**Key Findings:**

1. **Both models beat Buy & Hold**: In a -10.8% market decline, XGBoost returned +1.3% and CNN returned -0.24%
2. **XGBoost dominant**: Positive Sharpe (1.88) with meaningful alpha (+12.09%)
3. **CNN struggles with accuracy**: 47% accuracy but still protects capital
4. **High trade frequency**: 100 trades in 448 samples = ~22% active trading

**Why Models Outperform in Bear Market:**

1. **Bidirectional signals**: Models can profit from shorts during decline
2. **Class-weighted training**: Addresses 62% sell / 38% buy imbalance
3. **Liquidation features**: Capture market stress and potential reversals

**Limitations:**

1. **Bear market test period**: Results may differ in bull markets
2. **Single symbol**: BTCUSDT only, no diversification
3. **No leverage**: 10% position sizing is conservative

---

## 4. Model Architectures

### 4.1 XGBoost (Production Ready)

```python
xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=150,
    scale_pos_weight=1.61,  # Class imbalance
    device="cuda",
)
```

### 4.2 CNN for 2D Heatmap

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2)
        )
```

**Input**: (batch, 1, 50, 128) - 50 hours × 128 price bins

---

## 5. Conclusions

### What Works

- **Liquidation map features are predictive**: Cluster volume, concentration, and intensity have signal
- **XGBoost > CNN for this dataset**: Hand-crafted features outperform raw heatmaps
- **Bear market protection**: Both models protect capital during 10.8% decline

### What Doesn't Work

- **CNN accuracy**: 47% is near random; needs more data or better architecture
- **Buy precision**: 38% precision means many false positives

### Recommendations

1. **Use XGBoost for production**: Better accuracy, Sharpe, and interpretability
2. **Add more features**: Funding rate, orderbook imbalance
3. **Multi-symbol training**: ETHUSDT, SOLUSDT for diversification
4. **Walk-forward validation**: Rolling retraining for regime changes

---

## 6. Usage

### Download & Process Data

```bash
python -c "
import asyncio
from liquidation_map.ml.pipeline import DataPipeline

async def main():
    pipeline = DataPipeline()
    await pipeline.download_and_process('BTCUSDT', '2020-09-01', '2026-04-24')
    pipeline.close()

asyncio.run(main())
"
```

### Train XGBoost

```python
import xgboost as xgb
import polars as pl

df_train = pl.read_parquet("data/train/train_5y.parquet")
df_test = pl.read_parquet("data/train/test_2026.parquet")

model = xgb.XGBClassifier(
    max_depth=4, learning_rate=0.05, n_estimators=150,
    scale_pos_weight=1.61, device="cuda"
)
model.fit(X_train, y_train)
```

---

## Appendix: File Structure

```
data/
├── silver/                    # Hive-partitioned parquet
│   ├── dataset=klines/       # OHLCV data
│   └── dataset=open_interest/ # OI data
├── train/
│   ├── train_5y.parquet      # Training features
│   ├── test_2026.parquet     # Test features
│   ├── xgboost_5y.json       # Trained model
│   └── cnn_5y.pt             # Trained CNN

src/liquidation_map/ml/
├── pipeline.py      # Hive partitioned data pipeline
├── features.py      # Feature extraction (31 features)
├── labeling.py      # Triple-barrier labeling
├── backtest.py      # Backtest engine
└── models/
    ├── xgboost_model.py
    └── cnn_model.py
```
