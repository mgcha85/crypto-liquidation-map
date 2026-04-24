#!/usr/bin/env python3
"""Run multi-timeframe strategy comparison for BTC."""

import sys
sys.path.insert(0, "/mnt/data/projects/crypto-liquidation-map")

import json
from pathlib import Path
from datetime import datetime

from src.liquidation_map.ml.multi_timeframe import (
    MultiTimeframeStrategy,
    MultiTimeframeResults,
    TIMEFRAME_CONFIGS,
)


def run_single_timeframe_quick(
    symbol: str,
    timeframe: str,
    start_date: str = "2024-01-01",
    end_date: str = "2026-04-22",
    train_cutoff: str = "2025-12-31",
):
    strategy = MultiTimeframeStrategy(
        symbol=symbol,
        train_cutoff=train_cutoff,
    )
    return strategy.run_single_timeframe(timeframe, start_date, end_date)


def main():
    symbol = "BTCUSDT"
    output_dir = Path("data/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timeframes = ["1h", "15m", "5m"]
    all_results = {}
    
    for tf in timeframes:
        print(f"\n{'='*60}")
        print(f"Processing {tf} timeframe...")
        print(f"{'='*60}")
        
        try:
            result = run_single_timeframe_quick(symbol, tf)
            if result:
                all_results[tf] = {
                    "total_return": result.backtest.total_return,
                    "sharpe_ratio": result.backtest.sharpe_ratio,
                    "sortino_ratio": result.backtest.sortino_ratio,
                    "max_drawdown": result.backtest.max_drawdown,
                    "win_rate": result.backtest.win_rate,
                    "profit_factor": result.backtest.profit_factor,
                    "num_trades": result.backtest.num_trades,
                    "calmar_ratio": result.backtest.calmar_ratio,
                    "train_accuracy": result.train_accuracy,
                    "test_accuracy": result.test_accuracy,
                    "label_distribution": result.label_distribution,
                    "config": result.config,
                }
                print(f"\n{tf} Results:")
                print(f"  Return: {result.backtest.total_return*100:.2f}%")
                print(f"  Sharpe: {result.backtest.sharpe_ratio:.2f}")
                print(f"  Trades: {result.backtest.num_trades}")
        except Exception as e:
            print(f"Error processing {tf}: {e}")
            import traceback
            traceback.print_exc()
    
    output_path = output_dir / f"{symbol.lower()}_timeframe_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "symbol": symbol,
            "generated_at": datetime.now().isoformat(),
            "timeframes": all_results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Timeframe':<10} {'Return':>10} {'Sharpe':>10} {'Win Rate':>10} {'Trades':>10}")
    print("-" * 50)
    for tf, r in all_results.items():
        print(f"{tf:<10} {r['total_return']*100:>9.2f}% {r['sharpe_ratio']:>10.2f} {r['win_rate']*100:>9.1f}% {r['num_trades']:>10}")


if __name__ == "__main__":
    main()
