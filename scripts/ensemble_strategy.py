#!/usr/bin/env python3
"""Multi-timeframe ensemble: 5m signals filter 1h entries."""

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


def train_timeframe_model(
    symbol: str,
    timeframe: str,
    df_klines: pl.DataFrame,
    df_oi: pl.DataFrame,
    train_cutoff: str,
    horizon_bars: int,
    pt: float = 0.02,
    sl: float = 0.01,
) -> tuple:
    """Train a model for a specific timeframe."""
    strategy = MultiTimeframeStrategy(symbol, train_cutoff=train_cutoff)
    price_bucket = PRICE_BUCKETS.get(symbol, 250.0)
    feature_extractor = FeatureExtractor(price_bucket_size=price_bucket)
    
    barrier_config = BarrierConfig(profit_take=pt, stop_loss=sl, horizon=horizon_bars)
    labeler = TripleBarrierLabeler(barrier_config)
    df_labels = labeler.compute_labels(df_klines, return_details=True)
    
    timestamps = df_klines["timestamp"].to_list()
    lookback_hours = 50
    lookback_td = timedelta(hours=lookback_hours)
    
    rows = []
    step = max(1, horizon_bars // 4)
    
    for i in range(lookback_hours * 12, len(timestamps) - horizon_bars, step):
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
        return None, None, None
    
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
    
    if len(df_train_filtered) < 50 or len(df_test_filtered) < 20:
        return None, None, None
    
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
    
    return model, df_test_filtered, df_features


def run_ensemble_strategy(
    symbol: str = "BTCUSDT",
    start_date: str = "2025-06-01",
    end_date: str = "2026-04-22",
    train_cutoff: str = "2026-01-01",
):
    print("\n" + "="*60)
    print("MULTI-TIMEFRAME ENSEMBLE STRATEGY")
    print("5m signals filter 1h entries")
    print("="*60)
    
    loader = MultiTimeframeLoader()
    
    print("\nLoading 5m data...")
    df_5m = loader.get_candles(symbol, start_date, end_date, "5m")
    print(f"  {len(df_5m)} candles")
    
    print("Loading 1h data...")
    df_1h = loader.get_candles(symbol, start_date, end_date, "1h")
    print(f"  {len(df_1h)} candles")
    
    strategy = MultiTimeframeStrategy(symbol, train_cutoff=train_cutoff)
    df_oi = strategy.load_oi_data(start_date, end_date)
    print(f"Loaded {len(df_oi)} OI rows")
    
    print("\nTraining 5m model (filter signals)...")
    model_5m, df_test_5m, _ = train_timeframe_model(
        symbol, "5m", df_5m, df_oi, train_cutoff,
        horizon_bars=288, pt=0.02, sl=0.01
    )
    
    print("Training 1h model (entry signals)...")
    model_1h, df_test_1h, df_features_1h = train_timeframe_model(
        symbol, "1h", df_1h, df_oi, train_cutoff,
        horizon_bars=48, pt=0.02, sl=0.01
    )
    
    if model_5m is None or model_1h is None:
        print("Failed to train models!")
        return None
    
    print("\n--- Individual Model Performance ---")
    
    preds_5m = model_5m.predict(df_test_5m)
    proba_5m = model_5m.predict_proba(df_test_5m)
    signals_5m = np.where(preds_5m == 0, 1, -1)
    
    backtest_config = BacktestConfig(
        position_size_pct=0.1,
        taker_fee_pct=0.0004,
        slippage_bps=5.0,
    )
    backtester = Backtester(backtest_config)
    result_5m = backtester.run(df_test_5m, signals_5m, proba_5m)
    print(f"5m Model: Return={result_5m.total_return*100:.2f}%, Sharpe={result_5m.sharpe_ratio:.2f}, Trades={result_5m.num_trades}")
    
    preds_1h = model_1h.predict(df_test_1h)
    proba_1h = model_1h.predict_proba(df_test_1h)
    signals_1h = np.where(preds_1h == 0, 1, -1)
    
    result_1h = backtester.run(df_test_1h, signals_1h, proba_1h)
    print(f"1h Model: Return={result_1h.total_return*100:.2f}%, Sharpe={result_1h.sharpe_ratio:.2f}, Trades={result_1h.num_trades}")
    
    print("\n--- Ensemble Strategy ---")
    print("Rule: Only take 1h signals when 5m model agrees (same direction)")
    
    timestamps_5m = df_test_5m["timestamp"].to_list()
    signals_5m_list = list(signals_5m)
    
    timestamps_1h = df_test_1h["timestamp"].to_list()
    ensemble_signals = []
    ensemble_proba = []
    filtered_timestamps = []
    filtered_prices = []
    filtered_labels = []
    
    for i, ts_1h in enumerate(timestamps_1h):
        window_start = ts_1h - timedelta(hours=6)
        window_end = ts_1h + timedelta(minutes=30)
        
        recent_5m_signals = []
        for j, ts_5m in enumerate(timestamps_5m):
            if window_start <= ts_5m <= window_end:
                recent_5m_signals.append(signals_5m_list[j])
        
        signal_1h = signals_1h[i]
        
        if not recent_5m_signals:
            ensemble_signals.append(signal_1h)
            ensemble_proba.append(proba_1h[i])
            filtered_timestamps.append(ts_1h)
            filtered_prices.append(df_test_1h["current_price"][i])
            filtered_labels.append(df_test_1h["label"][i])
            continue
        
        avg_5m_signal = np.mean(recent_5m_signals)
        
        agreement = (signal_1h == 1 and avg_5m_signal > 0) or (signal_1h == -1 and avg_5m_signal < 0)
        
        if agreement:
            ensemble_signals.append(signal_1h)
            ensemble_proba.append(proba_1h[i])
            filtered_timestamps.append(ts_1h)
            filtered_prices.append(df_test_1h["current_price"][i])
            filtered_labels.append(df_test_1h["label"][i])
    
    print(f"Filtered {len(timestamps_1h)} → {len(ensemble_signals)} trades (5m agreement filter)")
    
    if len(ensemble_signals) < 5:
        print("Too few trades after filtering!")
        return None
    
    df_ensemble = pl.DataFrame({
        "timestamp": filtered_timestamps,
        "current_price": filtered_prices,
        "label": filtered_labels,
    })
    
    result_ensemble = backtester.run(
        df_ensemble,
        np.array(ensemble_signals),
        np.array(ensemble_proba)
    )
    
    prices = df_test_1h["current_price"].to_list()
    bh_return = (prices[-1] / prices[0] - 1) if prices else 0
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}")
    print("-" * 70)
    print(f"{'Buy & Hold':<25} {bh_return*100:>9.2f}% {'-':>8} {'-':>8} {'-':>8} {'-':>7}")
    print(f"{'5m Model':<25} {result_5m.total_return*100:>9.2f}% {result_5m.sharpe_ratio:>8.2f} {result_5m.max_drawdown*100:>7.2f}% {result_5m.win_rate*100:>7.1f}% {result_5m.num_trades:>7}")
    print(f"{'1h Model':<25} {result_1h.total_return*100:>9.2f}% {result_1h.sharpe_ratio:>8.2f} {result_1h.max_drawdown*100:>7.2f}% {result_1h.win_rate*100:>7.1f}% {result_1h.num_trades:>7}")
    print(f"{'Ensemble (5m→1h)':<25} {result_ensemble.total_return*100:>9.2f}% {result_ensemble.sharpe_ratio:>8.2f} {result_ensemble.max_drawdown*100:>7.2f}% {result_ensemble.win_rate*100:>7.1f}% {result_ensemble.num_trades:>7}")
    
    alpha_5m = result_5m.total_return - bh_return
    alpha_1h = result_1h.total_return - bh_return
    alpha_ensemble = result_ensemble.total_return - bh_return
    
    print(f"\nAlpha vs B&H:")
    print(f"  5m: {alpha_5m*100:+.2f}%")
    print(f"  1h: {alpha_1h*100:+.2f}%")
    print(f"  Ensemble: {alpha_ensemble*100:+.2f}%")
    
    output_dir = Path("data/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "symbol": symbol,
        "test_period": f"{train_cutoff} to {end_date}",
        "buy_hold_return": bh_return,
        "strategies": {
            "5m": {
                "total_return": result_5m.total_return,
                "sharpe_ratio": result_5m.sharpe_ratio,
                "max_drawdown": result_5m.max_drawdown,
                "win_rate": result_5m.win_rate,
                "profit_factor": result_5m.profit_factor,
                "num_trades": result_5m.num_trades,
                "alpha": alpha_5m,
            },
            "1h": {
                "total_return": result_1h.total_return,
                "sharpe_ratio": result_1h.sharpe_ratio,
                "max_drawdown": result_1h.max_drawdown,
                "win_rate": result_1h.win_rate,
                "profit_factor": result_1h.profit_factor,
                "num_trades": result_1h.num_trades,
                "alpha": alpha_1h,
            },
            "ensemble": {
                "total_return": result_ensemble.total_return,
                "sharpe_ratio": result_ensemble.sharpe_ratio,
                "max_drawdown": result_ensemble.max_drawdown,
                "win_rate": result_ensemble.win_rate,
                "profit_factor": result_ensemble.profit_factor,
                "num_trades": result_ensemble.num_trades,
                "alpha": alpha_ensemble,
                "filter_ratio": len(ensemble_signals) / len(timestamps_1h),
            },
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / f"{symbol.lower()}_ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_dir / f'{symbol.lower()}_ensemble_results.json'}")
    
    return results


if __name__ == "__main__":
    run_ensemble_strategy()
