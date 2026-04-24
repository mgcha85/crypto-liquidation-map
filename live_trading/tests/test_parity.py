import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import numpy as np

from live_trading.src.features import LiveFeatureExtractor
from live_trading.src.model import TradingModel
from live_trading.src.config import FEATURE_COLUMNS


def test_feature_parity():
    backtest_features_path = Path("checkpoints/backtest_features.json")
    if not backtest_features_path.exists():
        print("SKIP: No backtest features to compare")
        return True
    
    with open(backtest_features_path) as f:
        backtest_data = json.load(f)
    
    extractor = LiveFeatureExtractor()
    
    max_diff = 0.0
    mismatches = []
    
    for sample in backtest_data["samples"]:
        df_oi = pl.DataFrame(sample["oi_data"])
        df_klines = pl.DataFrame(sample["kline_data"])
        
        live_features = extractor.extract(
            df_oi,
            df_klines,
            sample["current_price"],
            sample["timestamp"],
        )
        
        for feat_name in FEATURE_COLUMNS:
            backtest_val = sample["features"].get(feat_name, 0.0)
            live_val = live_features.features.get(feat_name, 0.0)
            
            diff = abs(backtest_val - live_val)
            max_diff = max(max_diff, diff)
            
            if diff > 1e-6:
                mismatches.append({
                    "feature": feat_name,
                    "backtest": backtest_val,
                    "live": live_val,
                    "diff": diff,
                })
    
    print(f"Feature Parity Test")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mismatches (>1e-6): {len(mismatches)}")
    
    if mismatches:
        print("\nMismatch details:")
        for m in mismatches[:10]:
            print(f"  {m['feature']}: backtest={m['backtest']:.6f}, live={m['live']:.6f}, diff={m['diff']:.2e}")
    
    assert max_diff < 1e-6, f"Feature parity failed: max_diff={max_diff}"
    print("PASS: Feature parity verified")
    return True


def test_signal_parity():
    backtest_signals_path = Path("checkpoints/backtest_signals.json")
    if not backtest_signals_path.exists():
        print("SKIP: No backtest signals to compare")
        return True
    
    with open(backtest_signals_path) as f:
        backtest_data = json.load(f)
    
    model_path = Path("models/xgb_optuna_best.json")
    if not model_path.exists():
        print("SKIP: Model not found")
        return True
    
    model = TradingModel(model_path)
    extractor = LiveFeatureExtractor()
    
    matches = 0
    total = 0
    mismatches = []
    
    for sample in backtest_data["samples"]:
        df_oi = pl.DataFrame(sample["oi_data"])
        df_klines = pl.DataFrame(sample["kline_data"])
        
        features = extractor.extract(
            df_oi,
            df_klines,
            sample["current_price"],
            sample["timestamp"],
        )
        
        df_features = features.to_dataframe()
        live_signal = model.predict(df_features)
        backtest_signal = sample["signal"]
        
        total += 1
        if live_signal == backtest_signal:
            matches += 1
        else:
            mismatches.append({
                "timestamp": sample["timestamp"],
                "backtest": backtest_signal,
                "live": live_signal,
            })
    
    match_rate = matches / total if total > 0 else 0.0
    
    print(f"Signal Parity Test")
    print(f"  Total samples: {total}")
    print(f"  Matches: {matches}")
    print(f"  Match rate: {match_rate:.2%}")
    
    if mismatches:
        print(f"\nFirst 10 mismatches:")
        for m in mismatches[:10]:
            print(f"  {m['timestamp']}: backtest={m['backtest']}, live={m['live']}")
    
    assert match_rate == 1.0, f"Signal parity failed: match_rate={match_rate:.2%}"
    print("PASS: Signal parity verified")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("CP-001 & CP-002: Parity Verification")
    print("=" * 60)
    
    try:
        test_feature_parity()
        print()
        test_signal_parity()
        print()
        print("ALL PARITY TESTS PASSED")
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
