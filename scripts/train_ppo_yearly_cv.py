#!/usr/bin/env python3
"""
PPO training with yearly cross-validation.
Train on N-1 years, test on 1 year. Rotate test year.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
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


def train_ppo(env, policy, trainer, n_iterations: int = 50, n_steps: int = 2048) -> list:
    history = []
    
    for iteration in range(n_iterations):
        rollout = collect_rollout(env, policy, n_steps)
        
        advantages, returns = trainer.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        rollout["advantages"] = advantages
        rollout["returns"] = returns
        
        losses = trainer.update(rollout, n_epochs=10, batch_size=128)
        metrics = env.get_metrics()
        
        history.append({"iteration": iteration, **losses, **metrics})
        
        if iteration % 10 == 0:
            print(f"    Iter {iteration:3d} | Return: {metrics['total_return']*100:+.1f}% | "
                  f"Trades: {metrics['total_trades']} | WinRate: {metrics['win_rate']:.1%}")
    
    return history


def evaluate(env, policy) -> dict:
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _, _ = policy.get_action(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    return env.get_metrics()


def get_buy_hold_return(klines: pl.DataFrame, start_ts, end_ts) -> float:
    filtered = klines.filter(
        (pl.col("timestamp") >= start_ts) & (pl.col("timestamp") <= end_ts)
    ).sort("timestamp")
    
    if len(filtered) < 2:
        return 0.0
    
    start_price = filtered["close"][0]
    end_price = filtered["close"][-1]
    return (end_price - start_price) / start_price


def run_yearly_cv():
    print("Loading data...")
    klines, features = load_data()
    
    kline_ts = set(klines["timestamp"].to_list())
    feature_ts = set(features["timestamp"].to_list())
    common_ts = sorted(kline_ts & feature_ts)
    
    klines = klines.filter(pl.col("timestamp").is_in(common_ts))
    features = features.filter(pl.col("timestamp").is_in(common_ts))
    
    print(f"Total samples: {len(common_ts):,}")
    
    features = features.with_columns([
        pl.col("timestamp").dt.year().alias("year")
    ])
    years = sorted(features["year"].unique().to_list())
    print(f"Years: {years}")
    
    test_years = [2021, 2022, 2023, 2024, 2025]
    
    env_config = EnvConfig(
        initial_balance=100_000,
        position_size_pct=0.1,
        commission_rate=0.0004,
        slippage_bps=5.0,
        reward_scaling=0.1,
        inactivity_penalty=0.001,
        max_inactive_steps=30,
    )
    
    feature_config = RLFeatureConfig(
        candle_window=200,
        normalize=True,
    )
    
    results = []
    
    for test_year in test_years:
        print(f"\n{'='*60}")
        print(f"CV Fold: Test Year = {test_year}")
        print(f"{'='*60}")
        
        train_years = [y for y in years if y != test_year and y < test_year]
        if not train_years:
            print(f"  Skipping: no training years before {test_year}")
            continue
        
        print(f"  Train years: {train_years}")
        
        train_ts = features.filter(pl.col("year").is_in(train_years))["timestamp"].to_list()
        test_ts = features.filter(pl.col("year") == test_year)["timestamp"].to_list()
        
        klines_train = klines.filter(pl.col("timestamp").is_in(train_ts))
        klines_test = klines.filter(pl.col("timestamp").is_in(test_ts))
        features_train = features.filter(pl.col("timestamp").is_in(train_ts)).drop("year")
        features_test = features.filter(pl.col("timestamp") .is_in(test_ts)).drop("year")
        
        print(f"  Train: {len(klines_train):,} rows ({train_years[0]}-{train_years[-1]})")
        print(f"  Test: {len(klines_test):,} rows ({test_year})")
        
        if len(klines_train) < 1000 or len(klines_test) < 500:
            print(f"  Skipping: insufficient data")
            continue
        
        print(f"\n  Creating environment...")
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
        
        print(f"\n  Training PPO (50 iterations)...")
        train_history = train_ppo(train_env, policy, trainer, n_iterations=50, n_steps=2048)
        
        print(f"\n  Evaluating on {test_year}...")
        test_env = CryptoFuturesEnv(
            df_klines=klines_test,
            df_features=features_test,
            config=env_config,
            feature_config=feature_config,
        )
        
        test_metrics = evaluate(test_env, policy)
        
        bh_return = get_buy_hold_return(klines_test, test_ts[0], test_ts[-1])
        
        result = {
            "test_year": test_year,
            "train_years": train_years,
            "train_samples": len(klines_train),
            "test_samples": len(klines_test),
            "ppo_return": test_metrics["total_return"],
            "buy_hold_return": bh_return,
            "alpha": test_metrics["total_return"] - bh_return,
            "sharpe": test_metrics["sharpe_ratio"],
            "max_drawdown": test_metrics["max_drawdown"],
            "total_trades": test_metrics["total_trades"],
            "win_rate": test_metrics["win_rate"],
        }
        results.append(result)
        
        print(f"\n  === {test_year} Results ===")
        print(f"  PPO Return:      {result['ppo_return']*100:+.2f}%")
        print(f"  Buy & Hold:      {result['buy_hold_return']*100:+.2f}%")
        print(f"  Alpha:           {result['alpha']*100:+.2f}%")
        print(f"  Sharpe:          {result['sharpe']:.2f}")
        print(f"  Max Drawdown:    {result['max_drawdown']*100:.2f}%")
        print(f"  Trades:          {result['total_trades']}")
        print(f"  Win Rate:        {result['win_rate']:.1%}")
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n{'Year':<6} {'PPO':>10} {'B&H':>10} {'Alpha':>10} {'Sharpe':>8} {'Trades':>8} {'WinRate':>8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['test_year']:<6} "
              f"{r['ppo_return']*100:>+9.2f}% "
              f"{r['buy_hold_return']*100:>+9.2f}% "
              f"{r['alpha']*100:>+9.2f}% "
              f"{r['sharpe']:>8.2f} "
              f"{r['total_trades']:>8} "
              f"{r['win_rate']*100:>7.1f}%")
    
    if results:
        avg_ppo = np.mean([r["ppo_return"] for r in results])
        avg_bh = np.mean([r["buy_hold_return"] for r in results])
        avg_alpha = np.mean([r["alpha"] for r in results])
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_wr = np.mean([r["win_rate"] for r in results])
        
        print("-" * 70)
        print(f"{'AVG':<6} "
              f"{avg_ppo*100:>+9.2f}% "
              f"{avg_bh*100:>+9.2f}% "
              f"{avg_alpha*100:>+9.2f}% "
              f"{avg_sharpe:>8.2f} "
              f"{'':>8} "
              f"{avg_wr*100:>7.1f}%")
    
    output_path = Path("data/train/ppo_yearly_cv_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_yearly_cv()
