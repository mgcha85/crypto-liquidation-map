#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np

from src.liquidation_map.ml.features import FeatureExtractor
from src.liquidation_map.ml.multi_timeframe import MultiTimeframeLoader, MultiTimeframeStrategy, PRICE_BUCKETS
from src.liquidation_map.ml.labeling import TripleBarrierLabeler, BarrierConfig
from src.liquidation_map.ml.models.xgboost_model import XGBoostModel, XGBConfig
from live_trading.src.config import get_xgb_params_dict, FEATURE_COLUMNS


def build_training_data(
    symbol: str = "BTCUSDT",
    start_date: str = "2025-06-01",
    end_date: str = "2026-04-22",
) -> pl.DataFrame:
    loader = MultiTimeframeLoader()
    strategy = MultiTimeframeStrategy(symbol, train_cutoff="2026-01-01")
    price_bucket = PRICE_BUCKETS.get(symbol, 250.0)
    feature_extractor = FeatureExtractor(price_bucket_size=price_bucket)
    
    df_1h = loader.get_candles(symbol, start_date, end_date, "1h")
    df_oi = strategy.load_oi_data(start_date, end_date)
    
    barrier_config = BarrierConfig(profit_take=0.02, stop_loss=0.01, horizon=48)
    labeler = TripleBarrierLabeler(barrier_config)
    df_labels = labeler.compute_labels(df_1h, return_details=True)
    
    timestamps = df_1h["timestamp"].to_list()
    lookback_hours = 50
    lookback_td = timedelta(hours=lookback_hours)
    
    rows = []
    for i in range(lookback_hours, len(timestamps) - 48, 6):
        ts = timestamps[i]
        window_start = ts - lookback_td
        
        klines_window = df_1h.filter(
            (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
        )
        oi_window = df_oi.filter(
            (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
        )
        
        if len(oi_window) < 10:
            continue
        
        label_row = df_labels.filter(pl.col("timestamp") == ts)
        if label_row.is_empty():
            continue
        
        current_price = float(df_1h.filter(pl.col("timestamp") == ts)["close"][0])
        
        liq_features = feature_extractor.extract_window_features(oi_window, klines_window, current_price)
        candle_features = strategy._extract_candle_features_scaled(klines_window, 60)
        
        row = {
            "timestamp": ts,
            "close": current_price,
            "label": int(label_row["label"][0]),
            **liq_features,
            **candle_features,
        }
        rows.append(row)
    
    return pl.DataFrame(rows)


def train_model_with_sota_params(df_train: pl.DataFrame) -> XGBoostModel:
    import xgboost as xgb
    params = get_xgb_params_dict()
    
    df_train = df_train.with_columns([
        pl.when(pl.col("label") == -1).then(pl.lit(0))
        .otherwise(pl.lit(1)).alias("label_encoded")
    ])
    
    available_cols = [c for c in FEATURE_COLUMNS if c in df_train.columns]
    X = df_train.select(available_cols).to_numpy()
    y = df_train["label_encoded"].to_numpy()
    
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        tree_method=params["tree_method"],
        device=params["device"],
        random_state=params["random_state"],
    )
    model.fit(X, y, verbose=True)
    
    return model, available_cols


def generate_signals(model, feature_cols: list[str], df_test: pl.DataFrame) -> list[dict]:
    import xgboost as xgb
    signals = []
    
    X = df_test.select([c for c in feature_cols if c in df_test.columns]).to_numpy()
    dmatrix = xgb.DMatrix(X, feature_names=[c for c in feature_cols if c in df_test.columns])
    raw_preds = model.get_booster().predict(dmatrix)
    predictions = np.where(raw_preds > 0.5, 1, -1)
    
    for i, (row, pred) in enumerate(zip(df_test.iter_rows(named=True), predictions)):
        signals.append({
            "index": i,
            "timestamp": str(row.get("timestamp", "")),
            "close": row.get("close", 0.0),
            "signal": int(pred),
            "features": {col: row.get(col, 0.0) for col in FEATURE_COLUMNS if col in row},
        })
    
    return signals


def main():
    print("Building training data...")
    df = build_training_data()
    
    cutoff_dt = datetime(2026, 1, 1)
    df_train = df.filter(pl.col("timestamp") < cutoff_dt)
    df_test = df.filter(pl.col("timestamp") >= cutoff_dt)
    
    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")
    
    print("Training model with SOTA params...")
    model, feature_cols = train_model_with_sota_params(df_train)
    
    model_path = Path("live_trading/models/xgb_optuna_best.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"Model saved to: {model_path}")
    
    print("Generating signals...")
    signals = generate_signals(model, feature_cols, df_test)
    
    output = {
        "generated_at": datetime.now().isoformat(),
        "model_params": get_xgb_params_dict(),
        "test_samples": len(df_test),
        "samples": signals,
    }
    
    output_path = Path("live_trading/checkpoints/backtest_signals.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"Signals saved to: {output_path}")
    
    signal_counts = {-1: 0, 0: 0, 1: 0}
    for s in signals:
        signal_counts[s["signal"]] += 1
    
    print(f"\nSignal distribution:")
    print(f"  Sell (-1): {signal_counts[-1]}")
    print(f"  Hold (0):  {signal_counts[0]}")
    print(f"  Buy (+1):  {signal_counts[1]}")


if __name__ == "__main__":
    main()
