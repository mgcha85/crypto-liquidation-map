#!/usr/bin/env python3
"""Quick parametric study with fewer combinations."""

import sys
sys.path.insert(0, "/mnt/data/projects/crypto-liquidation-map")

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from src.liquidation_map.ml.multi_timeframe import (
    MultiTimeframeStrategy,
    MultiTimeframeLoader,
    PRICE_BUCKETS,
)
from src.liquidation_map.ml.features import FeatureExtractor
from src.liquidation_map.ml.labeling import TripleBarrierLabeler, BarrierConfig
from src.liquidation_map.ml.backtest import Backtester, BacktestConfig
from src.liquidation_map.ml.models.xgboost_model import XGBoostModel, XGBConfig


def quick_param_study(
    symbol: str = "BTCUSDT",
    start_date: str = "2025-06-01",
    end_date: str = "2026-04-22",
    train_cutoff: str = "2026-01-01",
) -> dict:
    print("\n" + "="*60)
    print("QUICK PARAMETRIC STUDY")
    print("="*60)
    
    configs = [
        {"pt": 0.015, "sl": 0.01, "horizon": 24, "name": "Conservative"},
        {"pt": 0.02, "sl": 0.01, "horizon": 24, "name": "Baseline"},
        {"pt": 0.025, "sl": 0.01, "horizon": 24, "name": "Aggressive PT"},
        {"pt": 0.02, "sl": 0.005, "horizon": 24, "name": "Tight SL"},
        {"pt": 0.02, "sl": 0.015, "horizon": 24, "name": "Wide SL"},
        {"pt": 0.02, "sl": 0.01, "horizon": 12, "name": "Short Horizon"},
        {"pt": 0.02, "sl": 0.01, "horizon": 48, "name": "Long Horizon"},
        {"pt": 0.03, "sl": 0.01, "horizon": 24, "name": "High R:R (3:1)"},
    ]
    
    loader = MultiTimeframeLoader()
    df_klines = loader.get_candles(symbol, start_date, end_date, "1h")
    
    strategy = MultiTimeframeStrategy(symbol, train_cutoff=train_cutoff)
    df_oi = strategy.load_oi_data(start_date, end_date)
    
    print(f"Loaded {len(df_klines)} klines, {len(df_oi)} OI rows")
    
    price_bucket = PRICE_BUCKETS.get(symbol, 250.0)
    feature_extractor = FeatureExtractor(price_bucket_size=price_bucket)
    
    results = []
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} (PT={cfg['pt']}, SL={cfg['sl']}, H={cfg['horizon']}) ---")
        
        barrier_config = BarrierConfig(
            profit_take=cfg["pt"],
            stop_loss=cfg["sl"],
            horizon=cfg["horizon"],
        )
        labeler = TripleBarrierLabeler(barrier_config)
        df_labels = labeler.compute_labels(df_klines, return_details=True)
        
        timestamps = df_klines["timestamp"].to_list()
        lookback_hours = 50
        lookback_td = timedelta(hours=lookback_hours)
        
        rows = []
        for i in range(lookback_hours, len(timestamps) - cfg["horizon"], 6):
            ts = timestamps[i]
            window_start = ts - lookback_td
            
            klines_window = df_klines.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            oi_window = df_oi.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            
            if len(oi_window) < 50:
                continue
            
            label_row = df_labels.filter(pl.col("timestamp") == ts)
            if label_row.is_empty():
                continue
            
            current_price = float(df_klines.filter(pl.col("timestamp") == ts)["close"][0])
            
            liq_features = feature_extractor.extract_window_features(
                oi_window, klines_window, current_price
            )
            candle_features = strategy._extract_candle_features_scaled(klines_window, 60)
            
            row = {
                "timestamp": ts,
                "current_price": current_price,
                "label": int(label_row["label"][0]),
                **liq_features,
                **candle_features,
            }
            rows.append(row)
        
        if len(rows) < 100:
            print(f"  Skipping: only {len(rows)} samples")
            continue
        
        df_features = pl.DataFrame(rows)
        
        cutoff = datetime.strptime(train_cutoff, "%Y-%m-%d")
        df_train = df_features.filter(pl.col("timestamp") < cutoff)
        df_test = df_features.filter(pl.col("timestamp") >= cutoff)
        
        df_train_filtered = df_train.filter(pl.col("label") != 0).with_columns([
            pl.when(pl.col("label") == -1).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("label")
        ])
        df_test_filtered = df_test.filter(pl.col("label") != 0).with_columns([
            pl.when(pl.col("label") == -1).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("label")
        ])
        
        print(f"  Train: {len(df_train_filtered)}, Test: {len(df_test_filtered)}")
        
        if len(df_train_filtered) < 50 or len(df_test_filtered) < 20:
            continue
        
        label_counts = df_train_filtered["label"].value_counts().to_dicts()
        sell_count = sum(r["count"] for r in label_counts if r["label"] == -1)
        buy_count = sum(r["count"] for r in label_counts if r["label"] == 0)
        scale_pos_weight = sell_count / max(buy_count, 1)
        
        xgb_config = XGBConfig(
            objective="binary:logistic",
            num_class=2,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
        
        model = XGBoostModel(xgb_config)
        model.train(df_train_filtered)
        
        test_preds = model.predict(df_test_filtered)
        test_proba = model.predict_proba(df_test_filtered)
        
        signals = np.where(test_preds == 0, 1, -1)
        
        backtest_config = BacktestConfig(
            position_size_pct=0.1,
            taker_fee_pct=0.0004,
            slippage_bps=5.0,
        )
        backtester = Backtester(backtest_config)
        backtest_result = backtester.run(df_test_filtered, signals, test_proba)
        
        result = {
            "name": cfg["name"],
            "profit_take": cfg["pt"],
            "stop_loss": cfg["sl"],
            "horizon": cfg["horizon"],
            "risk_reward": cfg["pt"] / cfg["sl"],
            "total_return": backtest_result.total_return,
            "sharpe_ratio": backtest_result.sharpe_ratio,
            "max_drawdown": backtest_result.max_drawdown,
            "win_rate": backtest_result.win_rate,
            "profit_factor": backtest_result.profit_factor,
            "num_trades": backtest_result.num_trades,
        }
        results.append(result)
        
        print(f"  Return: {backtest_result.total_return*100:.2f}%, Sharpe: {backtest_result.sharpe_ratio:.2f}, Trades: {backtest_result.num_trades}")
    
    output_dir = Path("data/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{symbol.lower()}_param_study.json", "w") as f:
        json.dump({
            "symbol": symbol,
            "generated_at": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)
    
    results_sorted = sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)
    
    print("\n" + "="*60)
    print("RESULTS RANKED BY SHARPE RATIO")
    print("="*60)
    print(f"{'Config':<20} {'R:R':>5} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}")
    print("-" * 75)
    for r in results_sorted:
        print(f"{r['name']:<20} {r['risk_reward']:>5.1f} "
              f"{r['total_return']*100:>9.2f}% {r['sharpe_ratio']:>8.2f} {r['max_drawdown']*100:>7.2f}% "
              f"{r['win_rate']*100:>7.1f}% {r['num_trades']:>7}")
    
    print(f"\nSaved to {output_dir / f'{symbol.lower()}_param_study.json'}")
    
    return {"results": results, "best": results_sorted[0] if results_sorted else None}


if __name__ == "__main__":
    quick_param_study()
