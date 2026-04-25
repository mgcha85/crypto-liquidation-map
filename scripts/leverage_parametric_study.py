#!/usr/bin/env python3
"""
Parametric study for optimal leverage and position size.
Grid search over leverage × position_size, evaluate with PPO on 2025 test data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import itertools
import numpy as np
import polars as pl
import torch
from datetime import datetime
from collections import defaultdict

from liquidation_map.rl import (
    CryptoFuturesEnv,
    EnvConfig,
    RLFeatureConfig,
    HybridTradingPolicy,
    PPOTrainer,
)


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    klines_path = Path("/mnt/data/finance/cryptocurrency/BTCUSDT")
    klines = pl.read_parquet(klines_path)
    klines = klines.sort("datetime")
    
    klines = klines.group_by_dynamic("datetime", every="1h").agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ])
    klines = klines.rename({"datetime": "timestamp"})
    
    features = pl.read_parquet("data/train/features_1h_full.parquet")
    
    return klines, features


def collect_rollout(env, policy, n_steps: int = 2048) -> dict:
    obs, _ = env.reset()
    rollout = defaultdict(list)
    
    for _ in range(n_steps):
        action, log_prob, value = policy.get_action(obs)
        
        rollout["candles"].append(obs["candles"])
        rollout["ml_features"].append(obs["ml_features"])
        rollout["portfolio"].append(obs["portfolio"])
        rollout["actions"].append(action)
        rollout["log_probs"].append(log_prob)
        rollout["values"].append(value)
        
        obs, reward, terminated, truncated, info = env.step(action)
        rollout["rewards"].append(reward)
        rollout["dones"].append(terminated or truncated)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    return {k: np.array(v) for k, v in rollout.items()}


def train_ppo(env, policy, trainer, n_iterations: int = 30, n_steps: int = 2048) -> list:
    for iteration in range(n_iterations):
        rollout = collect_rollout(env, policy, n_steps)
        
        advantages, returns = trainer.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        rollout["advantages"] = advantages
        rollout["returns"] = returns
        
        trainer.update(rollout, n_epochs=10, batch_size=128)
    
    return []


def evaluate(env, policy) -> dict:
    obs, _ = env.reset()
    done = False
    liquidated = False
    
    while not done:
        action, _, _ = policy.get_action(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if truncated and not terminated:
            liquidated = True
    
    metrics = env.get_metrics()
    metrics["liquidated"] = liquidated
    return metrics


def get_buy_hold_return(klines: pl.DataFrame, start_ts, end_ts) -> float:
    filtered = klines.filter(
        (pl.col("timestamp") >= start_ts) & (pl.col("timestamp") <= end_ts)
    ).sort("timestamp")
    
    if len(filtered) < 2:
        return 0.0
    
    start_price = filtered["close"][0]
    end_price = filtered["close"][-1]
    return (end_price - start_price) / start_price


def run_parametric_study():
    print("=" * 70)
    print("LEVERAGE × POSITION SIZE PARAMETRIC STUDY")
    print("=" * 70)
    
    leverage_values = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    position_size_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    print(f"\nLeverage values: {leverage_values}")
    print(f"Position size values: {position_size_values}")
    print(f"Total combinations: {len(leverage_values) * len(position_size_values)}")
    
    print("\nLoading data...")
    klines, features = load_data()
    
    kline_ts = set(klines["timestamp"].to_list())
    feature_ts = set(features["timestamp"].to_list())
    common_ts = sorted(kline_ts & feature_ts)
    
    klines = klines.filter(pl.col("timestamp").is_in(common_ts))
    features = features.filter(pl.col("timestamp").is_in(common_ts))
    
    features = features.with_columns([pl.col("timestamp").dt.year().alias("year")])
    
    train_years = [2020, 2021, 2022, 2023, 2024]
    test_year = 2025
    
    train_ts = features.filter(pl.col("year").is_in(train_years))["timestamp"].to_list()
    test_ts = features.filter(pl.col("year") == test_year)["timestamp"].to_list()
    
    klines_train = klines.filter(pl.col("timestamp").is_in(train_ts))
    klines_test = klines.filter(pl.col("timestamp").is_in(test_ts))
    features_train = features.filter(pl.col("timestamp").is_in(train_ts)).drop("year")
    features_test = features.filter(pl.col("timestamp").is_in(test_ts)).drop("year")
    
    print(f"\nTrain: {len(klines_train):,} rows ({train_years[0]}-{train_years[-1]})")
    print(f"Test: {len(klines_test):,} rows ({test_year})")
    
    bh_return = get_buy_hold_return(klines_test, test_ts[0], test_ts[-1])
    print(f"Buy & Hold return (2025): {bh_return*100:+.2f}%")
    
    feature_config = RLFeatureConfig(candle_window=200, normalize=True)
    
    results = []
    
    for leverage, position_size in itertools.product(leverage_values, position_size_values):
        print(f"\n--- Leverage: {leverage}x | Position Size: {position_size*100:.0f}% ---")
        
        env_config = EnvConfig(
            initial_balance=100_000,
            position_size_pct=position_size,
            leverage=leverage,
            commission_rate=0.0004,
            slippage_bps=5.0,
            reward_scaling=0.1,
            inactivity_penalty=0.001,
            max_inactive_steps=30,
            liquidation_threshold=0.9,
        )
        
        train_env = CryptoFuturesEnv(
            df_klines=klines_train,
            df_features=features_train,
            config=env_config,
            feature_config=feature_config,
        )
        
        ml_dim = train_env.feature_extractor.ml_feature_dim(features_train)
        candle_shape = train_env.feature_extractor.candle_shape
        
        policy = HybridTradingPolicy(
            candle_shape=candle_shape,
            ml_feature_dim=ml_dim,
            portfolio_dim=2,
            action_dim=3,
            hidden_dim=128,
        )
        
        trainer = PPOTrainer(
            policy=policy,
            lr=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            entropy_coef=0.05,
        )
        
        print("  Training (30 iterations)...", end=" ", flush=True)
        train_ppo(train_env, policy, trainer, n_iterations=30, n_steps=2048)
        print("Done")
        
        test_env = CryptoFuturesEnv(
            df_klines=klines_test,
            df_features=features_test,
            config=env_config,
            feature_config=feature_config,
        )
        
        test_metrics = evaluate(test_env, policy)
        
        result = {
            "leverage": leverage,
            "position_size_pct": position_size,
            "effective_exposure": leverage * position_size,
            "total_return": test_metrics["total_return"],
            "alpha": test_metrics["total_return"] - bh_return,
            "sharpe": test_metrics["sharpe_ratio"],
            "max_drawdown": test_metrics["max_drawdown"],
            "total_trades": test_metrics["total_trades"],
            "win_rate": test_metrics["win_rate"],
            "liquidated": test_metrics["liquidated"],
        }
        results.append(result)
        
        status = "⚠️ LIQUIDATED" if result["liquidated"] else ""
        print(f"  Return: {result['total_return']*100:+.2f}% | "
              f"Alpha: {result['alpha']*100:+.2f}% | "
              f"Sharpe: {result['sharpe']:.2f} | "
              f"MaxDD: {result['max_drawdown']*100:.2f}% | "
              f"WinRate: {result['win_rate']*100:.1f}% {status}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Ranked by Sharpe Ratio)")
    print("=" * 70)
    
    valid_results = [r for r in results if not r["liquidated"]]
    valid_results.sort(key=lambda x: x["sharpe"], reverse=True)
    
    print(f"\n{'Lev':>4} {'Pos%':>5} {'Exp':>5} {'Return':>10} {'Alpha':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("-" * 75)
    
    for r in valid_results[:15]:
        print(f"{r['leverage']:>4.0f}x {r['position_size_pct']*100:>4.0f}% "
              f"{r['effective_exposure']:>5.2f} "
              f"{r['total_return']*100:>+9.2f}% "
              f"{r['alpha']*100:>+9.2f}% "
              f"{r['sharpe']:>8.2f} "
              f"{r['max_drawdown']*100:>7.2f}% "
              f"{r['win_rate']*100:>7.1f}%")
    
    liquidated_count = sum(1 for r in results if r["liquidated"])
    print(f"\nLiquidated combinations: {liquidated_count}/{len(results)}")
    
    if valid_results:
        best = valid_results[0]
        print(f"\n🏆 OPTIMAL CONFIGURATION:")
        print(f"   Leverage: {best['leverage']}x")
        print(f"   Position Size: {best['position_size_pct']*100:.0f}%")
        print(f"   Effective Exposure: {best['effective_exposure']:.2f}x")
        print(f"   Expected Return: {best['total_return']*100:+.2f}%")
        print(f"   Alpha vs B&H: {best['alpha']*100:+.2f}%")
        print(f"   Sharpe Ratio: {best['sharpe']:.2f}")
    
    output_path = Path("data/train/leverage_parametric_study.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_year": test_year,
            "buy_hold_return": bh_return,
            "leverage_values": leverage_values,
            "position_size_values": position_size_values,
            "results": results,
            "optimal": valid_results[0] if valid_results else None,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_parametric_study()
