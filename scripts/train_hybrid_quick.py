#!/usr/bin/env python3
"""Quick hybrid model training with reduced data for faster iteration."""

import sys
sys.path.insert(0, "/mnt/data/projects/crypto-liquidation-map")

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.liquidation_map.ml.multi_timeframe import (
    MultiTimeframeStrategy,
    MultiTimeframeLoader,
    PRICE_BUCKETS,
)
from src.liquidation_map.ml.features import FeatureExtractor
from src.liquidation_map.ml.labeling import TripleBarrierLabeler, BarrierConfig
from src.liquidation_map.ml.backtest import Backtester, BacktestConfig
from src.liquidation_map.ml.models.hybrid_model import HybridModel, HybridConfig


def prepare_hybrid_data(
    symbol: str = "BTCUSDT",
    start_date: str = "2025-06-01",
    end_date: str = "2026-04-22",
    train_cutoff: str = "2026-01-01",
):
    """Prepare data for hybrid model training."""
    print("Loading data...")
    
    loader = MultiTimeframeLoader()
    df_klines = loader.get_candles(symbol, start_date, end_date, "1h")
    
    strategy = MultiTimeframeStrategy(symbol, train_cutoff=train_cutoff)
    df_oi = strategy.load_oi_data(start_date, end_date)
    
    print(f"Loaded {len(df_klines)} klines, {len(df_oi)} OI rows")
    
    barrier_config = BarrierConfig(profit_take=0.02, stop_loss=0.01, horizon=48)
    labeler = TripleBarrierLabeler(barrier_config)
    df_labels = labeler.compute_labels(df_klines, return_details=True)
    
    price_bucket = PRICE_BUCKETS.get(symbol, 250.0)
    feature_extractor = FeatureExtractor(price_bucket_size=price_bucket)
    
    timestamps = df_klines["timestamp"].to_list()
    lookback_hours = 100
    lookback_td = timedelta(hours=lookback_hours)
    
    candle_sequences = []
    liq_maps = []
    ml_features_list = []
    labels = []
    timestamps_out = []
    prices = []
    
    print("Building samples...")
    for i in range(lookback_hours, len(timestamps) - 48, 12):
        ts = timestamps[i]
        window_start = ts - lookback_td
        
        klines_window = df_klines.filter(
            (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
        )
        oi_window = df_oi.filter(
            (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
        )
        
        if len(klines_window) < 100 or len(oi_window) < 50:
            continue
        
        label_row = df_labels.filter(pl.col("timestamp") == ts)
        if label_row.is_empty():
            continue
        
        label = int(label_row["label"][0])
        if label == 0:
            continue
        
        current_price = float(df_klines.filter(pl.col("timestamp") == ts)["close"][0])
        
        candle_data = klines_window.select([
            "open", "high", "low", "close", "volume"
        ]).tail(100).to_numpy()
        
        if len(candle_data) < 100:
            continue
        
        candle_data_norm = candle_data.copy()
        candle_data_norm[:, :4] = candle_data_norm[:, :4] / current_price
        candle_data_norm[:, 4] = candle_data_norm[:, 4] / (candle_data_norm[:, 4].mean() + 1e-8)
        
        liq_map = feature_extractor.build_liquidation_heatmap(
            oi_window, current_price, n_buckets=50, time_buckets=100
        )
        
        ml_feats = feature_extractor.extract_window_features(
            oi_window, klines_window, current_price
        )
        candle_feats = strategy._extract_candle_features_scaled(klines_window, 60)
        all_feats = {**ml_feats, **candle_feats}
        
        feat_values = [float(v) if v is not None and not np.isnan(v) else 0.0 
                       for v in all_feats.values()]
        
        candle_sequences.append(candle_data_norm)
        liq_maps.append(liq_map)
        ml_features_list.append(feat_values)
        labels.append(0 if label == -1 else 1)
        timestamps_out.append(ts)
        prices.append(current_price)
    
    print(f"Built {len(labels)} samples")
    
    X_candles = np.array(candle_sequences, dtype=np.float32)
    X_liq = np.array(liq_maps, dtype=np.float32)
    X_ml = np.array(ml_features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    cutoff = datetime.strptime(train_cutoff, "%Y-%m-%d")
    train_mask = np.array([t < cutoff for t in timestamps_out])
    test_mask = ~train_mask
    
    return {
        "X_candles_train": X_candles[train_mask],
        "X_candles_test": X_candles[test_mask],
        "X_liq_train": X_liq[train_mask],
        "X_liq_test": X_liq[test_mask],
        "X_ml_train": X_ml[train_mask],
        "X_ml_test": X_ml[test_mask],
        "y_train": y[train_mask],
        "y_test": y[test_mask],
        "timestamps_test": [t for t, m in zip(timestamps_out, test_mask) if m],
        "prices_test": [p for p, m in zip(prices, test_mask) if m],
    }


def train_hybrid_model(data: dict, epochs: int = 30, batch_size: int = 32):
    """Train hybrid model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    X_candles_train = torch.tensor(data["X_candles_train"]).permute(0, 2, 1)
    X_liq_train = torch.tensor(data["X_liq_train"])
    X_ml_train = torch.tensor(data["X_ml_train"])
    y_train = torch.tensor(data["y_train"])
    
    X_candles_test = torch.tensor(data["X_candles_test"]).permute(0, 2, 1)
    X_liq_test = torch.tensor(data["X_liq_test"])
    X_ml_test = torch.tensor(data["X_ml_test"])
    y_test = torch.tensor(data["y_test"])
    
    train_dataset = TensorDataset(X_candles_train, X_liq_train, X_ml_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    config = HybridConfig(
        candle_length=100,
        candle_features=5,
        liq_map_length=100,
        liq_map_bins=50,
        ml_features=X_ml_train.shape[1],
        num_classes=2,
        dropout=0.3,
    )
    
    model = HybridModel(config).to(device)
    
    class_counts = np.bincount(data["y_train"])
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")
    print(f"Class distribution (train): {class_counts}")
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            candles, liq, ml, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(candles, liq, ml)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_candles = X_candles_test.to(device)
            test_liq = X_liq_test.to(device)
            test_ml = X_ml_test.to(device)
            test_labels = y_test.to(device)
            
            outputs = model(test_candles, test_liq, test_ml)
            _, predicted = outputs.max(1)
            val_acc = predicted.eq(test_labels).sum().item() / len(test_labels)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Train Acc={correct/total:.4f}, Val Acc={val_acc:.4f}")
    
    model.load_state_dict(best_model_state)
    
    return model, config


def backtest_hybrid(model, data: dict, device: torch.device):
    """Backtest hybrid model predictions."""
    model.eval()
    
    X_candles = torch.tensor(data["X_candles_test"]).permute(0, 2, 1).to(device)
    X_liq = torch.tensor(data["X_liq_test"]).to(device)
    X_ml = torch.tensor(data["X_ml_test"]).to(device)
    
    with torch.no_grad():
        outputs = model(X_candles, X_liq, X_ml)
        proba = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
    
    signals = np.where(preds == 1, 1, -1)
    
    df_test = pl.DataFrame({
        "timestamp": data["timestamps_test"],
        "current_price": data["prices_test"],
        "label": data["y_test"],
    })
    
    backtest_config = BacktestConfig(
        position_size_pct=0.1,
        taker_fee_pct=0.0004,
        slippage_bps=5.0,
    )
    backtester = Backtester(backtest_config)
    result = backtester.run(df_test, signals, proba)
    
    return result, preds, proba


def main():
    print("="*60)
    print("HYBRID MODEL TRAINING (Quick Version)")
    print("="*60)
    
    data = prepare_hybrid_data()
    
    if len(data["y_train"]) < 50:
        print("Not enough training data!")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = train_hybrid_model(data, epochs=30, batch_size=32)
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    result, preds, proba = backtest_hybrid(model, data, device)
    
    print(f"Total Return: {result.total_return*100:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown*100:.2f}%")
    print(f"Win Rate: {result.win_rate*100:.1f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Num Trades: {result.num_trades}")
    
    bh_return = (data["prices_test"][-1] / data["prices_test"][0] - 1)
    alpha = result.total_return - bh_return
    print(f"\nBuy & Hold: {bh_return*100:.2f}%")
    print(f"Alpha: {alpha*100:.2f}%")
    
    output_dir = Path("data/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": "Hybrid CNN-LSTM-MLP",
        "symbol": "BTCUSDT",
        "test_period": "2026-01-01 to 2026-04-22",
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "num_trades": result.num_trades,
        "buy_hold_return": bh_return,
        "alpha": alpha,
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "btcusdt_hybrid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    torch.save({
        "model_state": model.state_dict(),
        "config": config.__dict__,
    }, output_dir / "btcusdt_hybrid_model.pt")
    
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
