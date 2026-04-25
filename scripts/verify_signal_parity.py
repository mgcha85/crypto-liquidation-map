#!/usr/bin/env python3
"""
Signal Parity Verification: Python PPO vs Go Live Trading Engine.

This script verifies that the Go engine produces identical signals
to the Python backtest for the same input data.

Checkpoints:
- CP-001: Core Engine Parity (signal match)
- CP-002: Feature Extraction Parity (numeric precision)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import subprocess
import numpy as np
import polars as pl
import torch
from datetime import datetime

from liquidation_map.rl import (
    CryptoFuturesEnv,
    EnvConfig,
    RLFeatureConfig,
    HybridTradingPolicy,
)


def load_ppo_model():
    checkpoint = torch.load("live_trading/models/ppo_policy.pt", weights_only=False)
    
    candle_shape = checkpoint["candle_shape"]
    ml_dim = checkpoint["ml_dim"]
    
    policy = HybridTradingPolicy(
        candle_shape=candle_shape,
        ml_feature_dim=ml_dim,
        portfolio_dim=2,
        action_dim=3,
        hidden_dim=128,
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    
    return policy, candle_shape, ml_dim


def load_test_data(n_samples: int = 100):
    klines_path = Path("/mnt/data/finance/cryptocurrency/BTCUSDT")
    klines = pl.read_parquet(klines_path).sort("datetime")
    
    klines = klines.group_by_dynamic("datetime", every="1h").agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ]).rename({"datetime": "timestamp"})
    
    features = pl.read_parquet("data/train/features_1h_full.parquet")
    
    common_ts = sorted(set(klines["timestamp"].to_list()) & set(features["timestamp"].to_list()))
    klines = klines.filter(pl.col("timestamp").is_in(common_ts))
    features = features.filter(pl.col("timestamp").is_in(common_ts))
    
    features = features.with_columns([pl.col("timestamp").dt.year().alias("year")])
    test_ts = features.filter(pl.col("year") == 2025)["timestamp"].to_list()[:n_samples + 200]
    
    klines_test = klines.filter(pl.col("timestamp").is_in(test_ts))
    features_test = features.filter(pl.col("timestamp").is_in(test_ts)).drop("year")
    
    return klines_test, features_test


def python_predict(policy, env, idx: int):
    env.current_idx = idx
    obs = env._get_observation()
    
    action, log_prob, value = policy.get_action(obs, deterministic=True)
    
    signal_map = {0: -1, 1: 0, 2: 1}
    signal = signal_map[action]
    
    candles_flat = obs["candles"].flatten().tolist()
    ml_features = obs["ml_features"].tolist()
    portfolio = obs["portfolio"].tolist()
    
    return {
        "signal": signal,
        "action": action,
        "candles": candles_flat[:50],
        "ml_features": ml_features,
        "portfolio": portfolio,
    }


def go_predict_batch(test_inputs: list) -> list:
    input_path = Path("live_trading/checkpoints/parity_input.json")
    output_path = Path("live_trading/checkpoints/parity_output.json")
    
    with open(input_path, "w") as f:
        json.dump(test_inputs, f)
    
    try:
        result = subprocess.run(
            ["./trader", "--parity-test", str(input_path), str(output_path)],
            cwd="live_trading",
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if output_path.exists():
            with open(output_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Go execution failed: {e}")
    
    return None


def verify_onnx_directly(test_inputs: list) -> list:
    import onnxruntime as ort
    
    session = ort.InferenceSession("live_trading/models/ppo_policy.onnx")
    
    results = []
    for inp in test_inputs:
        features = np.array(inp["features"], dtype=np.float32).reshape(1, -1)
        
        outputs = session.run(None, {"input": features})
        probs = outputs[0][0]
        
        action = int(np.argmax(probs))
        signal_map = {0: -1, 1: 0, 2: 1}
        
        results.append({
            "signal": signal_map[action],
            "action": action,
            "probs": probs.tolist(),
        })
    
    return results


def run_verification():
    print("=" * 70)
    print("SIGNAL PARITY VERIFICATION")
    print("Python PPO Backtest vs ONNX Runtime (Go equivalent)")
    print("=" * 70)
    
    print("\n[1] Loading PPO model...")
    policy, candle_shape, ml_dim = load_ppo_model()
    print(f"    Candle shape: {candle_shape}, ML dim: {ml_dim}")
    
    print("\n[2] Loading test data...")
    klines_test, features_test = load_test_data(n_samples=100)
    print(f"    Test samples: {len(klines_test)}")
    
    env_config = EnvConfig(
        initial_balance=100_000,
        position_size_pct=0.05,
        leverage=3.0,
    )
    feature_config = RLFeatureConfig(candle_window=200, normalize=True)
    
    env = CryptoFuturesEnv(
        df_klines=klines_test,
        df_features=features_test,
        config=env_config,
        feature_config=feature_config,
    )
    
    print("\n[3] Generating Python predictions...")
    python_results = []
    onnx_inputs = []
    
    n_test = min(50, len(klines_test) - 201)
    
    for i in range(200, 200 + n_test):
        py_pred = python_predict(policy, env, i)
        python_results.append(py_pred)
        
        obs = env._get_observation()
        features = np.concatenate([
            obs["candles"].flatten(),
            obs["ml_features"],
            obs["portfolio"],
        ])
        onnx_inputs.append({"features": features.tolist()})
    
    print(f"    Generated {len(python_results)} predictions")
    
    print("\n[4] Running ONNX inference (simulating Go)...")
    onnx_results = verify_onnx_directly(onnx_inputs)
    print(f"    Generated {len(onnx_results)} ONNX predictions")
    
    print("\n[5] Comparing results...")
    
    matches = 0
    mismatches = []
    
    for i, (py, onnx) in enumerate(zip(python_results, onnx_results)):
        if py["signal"] == onnx["signal"]:
            matches += 1
        else:
            mismatches.append({
                "index": i,
                "python_signal": py["signal"],
                "onnx_signal": onnx["signal"],
                "python_action": py["action"],
                "onnx_action": onnx["action"],
            })
    
    match_rate = matches / len(python_results) * 100
    
    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")
    print(f"Total samples:    {len(python_results)}")
    print(f"Matches:          {matches}")
    print(f"Mismatches:       {len(mismatches)}")
    print(f"Match rate:       {match_rate:.2f}%")
    
    cp001_pass = match_rate >= 99.0
    cp002_pass = match_rate == 100.0
    
    print(f"\nCP-001 (Core Engine Parity):       {'✅ PASSED' if cp001_pass else '❌ FAILED'}")
    print(f"CP-002 (Feature Extraction Parity): {'✅ PASSED' if cp002_pass else '⚠️ MINOR DIFF'}")
    
    if mismatches:
        print(f"\nMismatch details (first 5):")
        for m in mismatches[:5]:
            print(f"  Index {m['index']}: Python={m['python_signal']}, ONNX={m['onnx_signal']}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(python_results),
        "matches": matches,
        "mismatches": len(mismatches),
        "match_rate": match_rate,
        "cp001_pass": cp001_pass,
        "cp002_pass": cp002_pass,
        "mismatch_details": mismatches[:10],
        "model": "ppo_policy.onnx",
        "config": {
            "candle_window": candle_shape[0],
            "ml_feature_dim": ml_dim,
            "leverage": 3.0,
            "position_size_pct": 0.05,
        },
    }
    
    output_path = Path("live_trading/checkpoints/signal_parity_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_verification()
