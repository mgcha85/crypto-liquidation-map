#!/usr/bin/env python3
"""Export XGBoost model to ONNX for Go inference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


def main():
    model_path = Path("../models/xgb_optuna_best.json")
    if not model_path.exists():
        model_path = Path("data/train/btcusdt_xgboost.json")
    
    if not model_path.exists():
        print("Model file not found. Training new model...")
        train_and_export()
        return
    
    print(f"Loading model from {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    export_to_onnx(model)


def train_and_export():
    from datetime import datetime, timedelta
    import polars as pl
    
    from src.liquidation_map.ml.multi_timeframe import MultiTimeframeLoader, MultiTimeframeStrategy, PRICE_BUCKETS
    from src.liquidation_map.ml.features import FeatureExtractor
    from src.liquidation_map.ml.labeling import TripleBarrierLabeler, BarrierConfig
    
    symbol = "BTCUSDT"
    loader = MultiTimeframeLoader()
    strategy = MultiTimeframeStrategy(symbol, train_cutoff="2026-01-01")
    feature_extractor = FeatureExtractor(price_bucket_size=PRICE_BUCKETS.get(symbol, 250.0))
    
    print("Loading data...")
    df_1h = loader.get_candles(symbol, "2025-06-01", "2026-04-22", "1h")
    df_oi = strategy.load_oi_data("2025-06-01", "2026-04-22")
    
    barrier_config = BarrierConfig(profit_take=0.02, stop_loss=0.01, horizon=48)
    labeler = TripleBarrierLabeler(barrier_config)
    df_labels = labeler.compute_labels(df_1h, return_details=True)
    
    print("Building features...")
    timestamps = df_1h["timestamp"].to_list()
    lookback_td = timedelta(hours=50)
    
    rows = []
    for i in range(50, len(timestamps) - 48, 6):
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
    
    df = pl.DataFrame(rows)
    cutoff_dt = datetime(2026, 1, 1)
    df_train = df.filter(pl.col("timestamp") < cutoff_dt)
    
    feature_cols = [
        "total_intensity", "long_intensity", "short_intensity", "long_short_ratio",
        "above_below_ratio", "near_1pct_concentration", "near_2pct_concentration",
        "near_5pct_concentration", "largest_long_cluster_distance", "largest_short_cluster_distance",
        "largest_long_cluster_volume", "largest_short_cluster_volume",
        "top3_long_dist_1", "top3_long_dist_2", "top3_long_dist_3",
        "top3_short_dist_1", "top3_short_dist_2", "top3_short_dist_3",
        "entropy", "skewness",
        "return_1h", "return_6h", "return_12h", "return_24h",
        "volatility_6h", "volatility_24h", "atr_24h", "volume_ma_ratio",
        "wick_ratio_upper", "wick_ratio_lower", "price_position",
    ]
    
    available_cols = [c for c in feature_cols if c in df_train.columns]
    X = df_train.select(available_cols).to_numpy()
    
    y = df_train["label"].to_numpy()
    y = np.where(y == -1, 0, 1)
    
    print("Training model with SOTA params...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=8,
        learning_rate=0.0137,
        n_estimators=175,
        subsample=0.681,
        colsample_bytree=0.668,
        gamma=1.183,
        reg_alpha=2.367,
        reg_lambda=1.127,
        tree_method="hist",
        device="cpu",
        random_state=42,
    )
    model.fit(X, y)
    
    model.save_model("live_trading/models/xgb_optuna_best.json")
    print("Saved JSON model")
    
    export_to_onnx(model)


def export_to_onnx(model):
    initial_type = [("input", FloatTensorType([None, 31]))]
    
    print("Converting to ONNX...")
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    
    output_path = Path("live_trading/models/xgb_optuna_best.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"ONNX model saved to {output_path}")


if __name__ == "__main__":
    main()
