#!/usr/bin/env python3
"""
Full backtest script for liquidation map trading strategy.

Usage:
    python scripts/run_backtest.py --symbol BTCUSDT --start 2020-09-01 --end 2026-04-24
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import polars as pl

from liquidation_map.ml.pipeline import DataPipeline
from liquidation_map.ml.dataset import TrainingDataGenerator, WindowConfig
from liquidation_map.ml.labeling import BarrierConfig
from liquidation_map.ml.backtest import Backtester, BacktestConfig, WalkForwardValidator
from liquidation_map.ml.models.xgboost_model import XGBoostModel, XGBConfig, evaluate_classification


def run_backtest(
    symbol: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    use_gpu: bool = True,
):
    pipeline = DataPipeline(
        raw_dir="data/raw",
        silver_dir="data/silver",
        metadata_db="data/metadata.duckdb",
    )
    
    window_config = WindowConfig(
        lookback_hours=50,
        horizon_hours=24,
        step_hours=1,
        min_oi_rows=100,
    )
    
    barrier_config = BarrierConfig(
        profit_take=0.02,
        stop_loss=0.01,
        horizon=24,
    )
    
    generator = TrainingDataGenerator(
        pipeline=pipeline,
        window_config=window_config,
        barrier_config=barrier_config,
        price_bucket_size=250,
    )
    
    print(f"\n=== Generating Training Data ({train_start} ~ {train_end}) ===")
    df_train = generator.generate_dataset(
        symbol=symbol,
        start_date=train_start,
        end_date=train_end,
        output_path=Path(f"data/train/{symbol}_train.parquet"),
    )
    
    print(f"\n=== Generating Test Data ({test_start} ~ {test_end}) ===")
    df_test = generator.generate_dataset(
        symbol=symbol,
        start_date=test_start,
        end_date=test_end,
        output_path=Path(f"data/train/{symbol}_test.parquet"),
    )
    
    if df_train.is_empty() or df_test.is_empty():
        print("Error: Not enough data for training/testing")
        return
    
    print(f"\n=== Dataset Summary ===")
    print(f"Train samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    
    train_labels = df_train["label"].value_counts().sort("label")
    test_labels = df_test["label"].value_counts().sort("label")
    print(f"Train label distribution: {train_labels.to_dict()}")
    print(f"Test label distribution: {test_labels.to_dict()}")
    
    print(f"\n=== Training XGBoost Model ===")
    xgb_config = XGBConfig(
        device="cuda" if use_gpu else "cpu",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        early_stopping_rounds=20,
    )
    
    split_idx = int(len(df_train) * 0.8)
    df_train_split = df_train[:split_idx]
    df_val_split = df_train[split_idx:]
    
    model = XGBoostModel(xgb_config)
    train_result = model.train(df_train_split, df_val_split)
    print(f"Training result: {train_result}")
    
    print(f"\n=== Feature Importance (Top 10) ===")
    importance = model.get_feature_importance()
    print(importance.head(10))
    
    print(f"\n=== Test Set Evaluation ===")
    test_preds = model.predict(df_test)
    test_proba = model.predict_proba(df_test)
    test_labels_arr = df_test["label"].to_numpy()
    
    metrics = evaluate_classification(test_labels_arr, test_preds, test_proba)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (buy): {metrics['precision_buy']:.4f}")
    print(f"Recall (buy): {metrics['recall_buy']:.4f}")
    print(f"F1 (buy): {metrics['f1_buy']:.4f}")
    
    print(f"\n=== Backtest on Test Period ===")
    backtest_config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.1,
        taker_fee_pct=0.0004,
        slippage_bps=5,
    )
    
    backtester = Backtester(backtest_config)
    result = backtester.run(df_test, test_preds, test_proba)
    
    print(f"\n=== Backtest Results ===")
    print(f"Total Return: {result.total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"Calmar Ratio: {result.calmar_ratio:.2f}")
    print(f"Win Rate: {result.win_rate * 100:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Number of Trades: {result.num_trades}")
    print(f"Avg Trade PnL: ${result.avg_trade_pnl:.2f}")
    
    model.save(f"models/{symbol}_xgboost.json")
    print(f"\nModel saved to models/{symbol}_xgboost.json")
    
    pipeline.close()
    
    return {
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "accuracy": metrics["accuracy"],
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "num_trades": result.num_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Run liquidation map backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--train-start", default="2020-09-01", help="Training start date")
    parser.add_argument("--train-end", default="2025-12-31", help="Training end date")
    parser.add_argument("--test-start", default="2026-01-01", help="Test start date")
    parser.add_argument("--test-end", default="2026-04-24", help="Test end date")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    
    args = parser.parse_args()
    
    results = run_backtest(
        symbol=args.symbol,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        use_gpu=not args.no_gpu,
    )
    
    if results:
        print(f"\n=== Summary ===")
        print(f"2026 Test Return: {results['total_return'] * 100:.2f}%")
        print(f"2026 Sharpe: {results['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()
