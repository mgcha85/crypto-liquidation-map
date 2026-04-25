#!/usr/bin/env python3
"""
Generate 1-hour interval features for full 6-year dataset.
Uses klines from /mnt/data/finance/cryptocurrency and OI from data/silver.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import numpy as np

from liquidation_map.ml.features import FeatureExtractor
from liquidation_map.ml.labeling import TripleBarrierLabeler, BarrierConfig


def load_klines_1h(symbol: str = "BTCUSDT") -> pl.DataFrame:
    path = Path(f"/mnt/data/finance/cryptocurrency/{symbol}")
    df = pl.read_parquet(path)
    df = df.sort("datetime")
    
    df_1h = df.group_by_dynamic("datetime", every="1h").agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
        pl.col("quote_volume").sum(),
        pl.col("taker_buy_base").sum(),
        pl.col("taker_buy_quote").sum(),
    ])
    
    df_1h = df_1h.rename({"datetime": "timestamp"})
    return df_1h


def load_oi(symbol: str = "BTCUSDT") -> pl.DataFrame:
    path = Path(f"data/silver/dataset=open_interest/symbol={symbol}")
    if not path.exists():
        print(f"OI data not found: {path}")
        return pl.DataFrame()
    
    df = pl.read_parquet(path)
    df = df.sort("timestamp")
    return df


def generate_features_1h(
    symbol: str = "BTCUSDT",
    lookback_hours: int = 50,
    horizon_hours: int = 24,
    output_path: str = None,
) -> pl.DataFrame:
    print(f"Loading klines for {symbol}...")
    df_klines = load_klines_1h(symbol)
    print(f"  Loaded {len(df_klines):,} 1h candles")
    print(f"  Range: {df_klines['timestamp'].min()} to {df_klines['timestamp'].max()}")
    
    print(f"\nLoading OI data...")
    df_oi = load_oi(symbol)
    if df_oi.is_empty():
        print("  No OI data available, using klines-only features")
        use_oi = False
    else:
        print(f"  Loaded {len(df_oi):,} OI rows")
        print(f"  Range: {df_oi['timestamp'].min()} to {df_oi['timestamp'].max()}")
        use_oi = True
    
    barrier_config = BarrierConfig(
        profit_take=0.02,
        stop_loss=0.01,
        horizon=horizon_hours,
    )
    labeler = TripleBarrierLabeler(barrier_config)
    feature_extractor = FeatureExtractor(price_bucket_size=100.0)
    
    print(f"\nComputing labels...")
    df_labels = labeler.compute_labels(df_klines, return_details=True)
    print(f"  Generated {len(df_labels):,} labels")
    
    print(f"\nGenerating features (1h interval)...")
    rows = []
    timestamps = df_klines["timestamp"].to_list()
    lookback = timedelta(hours=lookback_hours)
    
    valid_start_idx = lookback_hours
    valid_end_idx = len(timestamps) - horizon_hours
    
    total_steps = valid_end_idx - valid_start_idx
    print(f"  Processing {total_steps:,} samples...")
    
    for i in range(valid_start_idx, valid_end_idx):
        ts = timestamps[i]
        window_start = ts - lookback
        
        klines_window = df_klines.filter(
            (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
        )
        
        if len(klines_window) < 10:
            continue
        
        current_price = float(df_klines.filter(pl.col("timestamp") == ts)["close"][0])
        
        if use_oi:
            oi_window = df_oi.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            if len(oi_window) >= 10:
                liq_features = feature_extractor.extract_window_features(
                    oi_window, klines_window, current_price
                )
            else:
                liq_features = feature_extractor._empty_features(current_price)
        else:
            liq_features = feature_extractor._empty_features(current_price)
        
        candle_features = feature_extractor.extract_candle_features(klines_window)
        
        label_row = df_labels.filter(pl.col("timestamp") == ts)
        if label_row.is_empty():
            continue
        
        row = {
            "timestamp": ts,
            "current_price": current_price,
            **liq_features,
            **candle_features,
            "label": int(label_row["label"][0]),
            "mfe": float(label_row["mfe"][0]),
            "mae": float(label_row["mae"][0]),
        }
        rows.append(row)
        
        if len(rows) % 5000 == 0:
            print(f"    Processed {len(rows):,} / {total_steps:,} samples...")
    
    print(f"  Generated {len(rows):,} feature rows")
    
    if not rows:
        return pl.DataFrame()
    
    df_features = pl.DataFrame(rows)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.write_parquet(output_path)
        print(f"\nSaved to {output_path}")
    
    return df_features


def main():
    df = generate_features_1h(
        symbol="BTCUSDT",
        lookback_hours=50,
        horizon_hours=24,
        output_path="data/train/features_1h_full.parquet",
    )
    
    if df.is_empty():
        print("No features generated!")
        return
    
    print(f"\n=== Dataset Summary ===")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Columns: {len(df.columns)}")
    
    df = df.with_columns([
        pl.col("timestamp").dt.year().alias("year")
    ])
    print(f"\nRows per year:")
    print(df.group_by("year").len().sort("year"))


if __name__ == "__main__":
    main()
