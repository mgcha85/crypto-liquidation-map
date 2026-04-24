#!/usr/bin/env python3
"""Run all BTC strategies and generate comparison data for GitHub Pages."""

import sys
sys.path.insert(0, "/mnt/data/projects/crypto-liquidation-map")

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from src.liquidation_map.ml.multi_timeframe import (
    MultiTimeframeStrategy,
    MultiTimeframeLoader,
    TIMEFRAME_CONFIGS,
)
from src.liquidation_map.ml.backtest import Backtester, BacktestConfig


def run_timeframe_strategies(
    symbol: str = "BTCUSDT",
    start_date: str = "2025-01-01",
    end_date: str = "2026-04-22",
    train_cutoff: str = "2025-12-31",
) -> dict:
    print("\n" + "="*60)
    print("MULTI-TIMEFRAME STRATEGIES")
    print("="*60)
    
    strategy = MultiTimeframeStrategy(symbol, train_cutoff=train_cutoff)
    results = {}
    
    for tf in ["5m", "15m", "1h"]:
        print(f"\n--- {tf} ---")
        try:
            result = strategy.run_single_timeframe(tf, start_date, end_date)
            if result:
                results[tf] = {
                    "total_return": result.backtest.total_return,
                    "sharpe_ratio": result.backtest.sharpe_ratio,
                    "sortino_ratio": result.backtest.sortino_ratio,
                    "max_drawdown": result.backtest.max_drawdown,
                    "calmar_ratio": result.backtest.calmar_ratio,
                    "win_rate": result.backtest.win_rate,
                    "profit_factor": result.backtest.profit_factor,
                    "num_trades": result.backtest.num_trades,
                    "train_accuracy": result.train_accuracy,
                    "test_accuracy": result.test_accuracy,
                }
        except Exception as e:
            print(f"Error: {e}")
    
    return results


def calculate_buy_hold_return(
    symbol: str,
    start_date: str,
    end_date: str,
    train_cutoff: str,
) -> float:
    loader = MultiTimeframeLoader()
    df = loader.get_candles(symbol, start_date, end_date, "1h")
    
    cutoff = datetime.strptime(train_cutoff, "%Y-%m-%d")
    df_test = df.filter(pl.col("timestamp") >= cutoff)
    
    if df_test.is_empty():
        return 0.0
    
    start_price = df_test["close"].to_list()[0]
    end_price = df_test["close"].to_list()[-1]
    return (end_price - start_price) / start_price


def main():
    symbol = "BTCUSDT"
    start_date = "2025-01-01"
    end_date = "2026-04-22"
    train_cutoff = "2025-12-31"
    
    output_dir = Path("data/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timeframe_results = run_timeframe_strategies(
        symbol, start_date, end_date, train_cutoff
    )
    
    buy_hold = calculate_buy_hold_return(symbol, start_date, end_date, train_cutoff)
    
    all_results = {
        "symbol": symbol,
        "test_period": {
            "start": train_cutoff,
            "end": end_date,
        },
        "buy_hold_return": buy_hold,
        "generated_at": datetime.now().isoformat(),
        "strategies": {
            "timeframes": timeframe_results,
        }
    }
    
    output_path = output_dir / f"{symbol.lower()}_all_strategies.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*60)
    print(f"Buy & Hold Return: {buy_hold*100:.2f}%")
    print()
    print(f"{'Strategy':<15} {'Return':>10} {'Alpha':>10} {'Sharpe':>10} {'Win Rate':>10} {'Trades':>10}")
    print("-" * 65)
    
    for tf, r in timeframe_results.items():
        alpha = r["total_return"] - buy_hold
        print(f"XGB-{tf:<10} {r['total_return']*100:>9.2f}% {alpha*100:>9.2f}% {r['sharpe_ratio']:>10.2f} {r['win_rate']*100:>9.1f}% {r['num_trades']:>10}")


if __name__ == "__main__":
    main()
