#!/usr/bin/env python3
"""
Export trained PPO policy to ONNX format for Go inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from collections import defaultdict

from liquidation_map.rl import (
    CryptoFuturesEnv,
    EnvConfig,
    RLFeatureConfig,
    HybridTradingPolicy,
    PPOTrainer,
)


class PPOPolicyForExport(nn.Module):
    """Wrapper for ONNX export with single tensor input."""
    
    def __init__(self, policy: HybridTradingPolicy, candle_window: int, ml_feature_dim: int):
        super().__init__()
        self.policy = policy
        self.candle_window = candle_window
        self.ml_feature_dim = ml_feature_dim
        self.candle_features = 5
        self.portfolio_dim = 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candle_size = self.candle_window * self.candle_features
        
        candles = x[:, :candle_size].reshape(-1, self.candle_window, self.candle_features)
        ml_features = x[:, candle_size:candle_size + self.ml_feature_dim]
        portfolio = x[:, candle_size + self.ml_feature_dim:]
        
        logits, _ = self.policy(candles, ml_features, portfolio)
        probs = torch.softmax(logits, dim=-1)
        
        return probs


def load_data():
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


def train_ppo(env, policy, trainer, n_iterations: int = 50, n_steps: int = 2048):
    print(f"Training PPO ({n_iterations} iterations)...")
    
    for iteration in range(n_iterations):
        rollout = collect_rollout(env, policy, n_steps)
        
        advantages, returns = trainer.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        rollout["advantages"] = advantages
        rollout["returns"] = returns
        
        losses = trainer.update(rollout, n_epochs=10, batch_size=128)
        
        if iteration % 10 == 0:
            metrics = env.get_metrics()
            print(f"  Iter {iteration:3d} | Return: {metrics['total_return']*100:+.1f}% | "
                  f"Trades: {metrics['total_trades']} | WinRate: {metrics['win_rate']:.1%}")


def main():
    print("=" * 60)
    print("EXPORT PPO POLICY TO ONNX")
    print("=" * 60)
    
    print("\nLoading data...")
    klines, features = load_data()
    
    kline_ts = set(klines["timestamp"].to_list())
    feature_ts = set(features["timestamp"].to_list())
    common_ts = sorted(kline_ts & feature_ts)
    
    klines = klines.filter(pl.col("timestamp").is_in(common_ts))
    features = features.filter(pl.col("timestamp").is_in(common_ts))
    
    features = features.with_columns([pl.col("timestamp").dt.year().alias("year")])
    train_years = [2020, 2021, 2022, 2023, 2024]
    train_ts = features.filter(pl.col("year").is_in(train_years))["timestamp"].to_list()
    
    klines_train = klines.filter(pl.col("timestamp").is_in(train_ts))
    features_train = features.filter(pl.col("timestamp").is_in(train_ts)).drop("year")
    
    print(f"Train data: {len(klines_train):,} rows (2020-2024)")
    
    env_config = EnvConfig(
        initial_balance=100_000,
        position_size_pct=0.25,
        leverage=2.0,
        stop_loss_pct=0.05,
        commission_rate=0.0004,
        slippage_bps=5.0,
        reward_scaling=0.1,
        inactivity_penalty=0.001,
        max_inactive_steps=30,
    )
    
    feature_config = RLFeatureConfig(candle_window=200, normalize=True)
    
    print("\nCreating environment...")
    train_env = CryptoFuturesEnv(
        df_klines=klines_train,
        df_features=features_train,
        config=env_config,
        feature_config=feature_config,
    )
    
    ml_dim = train_env.feature_extractor.ml_feature_dim(features_train)
    candle_shape = train_env.feature_extractor.candle_shape
    
    print(f"Candle shape: {candle_shape}")
    print(f"ML feature dim: {ml_dim}")
    
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
    
    print()
    train_ppo(train_env, policy, trainer, n_iterations=30, n_steps=1024)
    
    output_dir = Path("live_trading/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch_path = output_dir / "ppo_policy.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "candle_shape": candle_shape,
        "ml_feature_dim": ml_dim,
        "portfolio_dim": 2,
        "action_dim": 3,
        "hidden_dim": 128,
        "env_config": {
            "leverage": env_config.leverage,
            "position_size_pct": env_config.position_size_pct,
            "stop_loss_pct": env_config.stop_loss_pct,
            "commission_rate": env_config.commission_rate,
            "slippage_bps": env_config.slippage_bps,
        },
    }, torch_path)
    print(f"\nSaved PyTorch model: {torch_path}")
    
    print("\nExporting to ONNX...")
    
    candle_window = candle_shape[0]
    candle_features = candle_shape[1]
    
    export_model = PPOPolicyForExport(policy, candle_window, ml_dim)
    export_model.eval()
    
    input_dim = candle_window * candle_features + ml_dim + 2
    dummy_input = torch.randn(1, input_dim)
    
    onnx_path = output_dir / "ppo_policy.onnx"
    
    torch.onnx.export(
        export_model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["action_probs"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "action_probs": {0: "batch_size"},
        },
    )
    
    print(f"Saved ONNX model: {onnx_path}")
    
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation: PASSED")
    
    config_path = output_dir / "ppo_config.json"
    config = {
        "model_type": "ppo_hybrid",
        "input_dim": input_dim,
        "candle_window": candle_window,
        "candle_features": candle_features,
        "ml_feature_dim": ml_dim,
        "portfolio_dim": 2,
        "action_dim": 3,
        "actions": ["short", "hold", "long"],
        "trading_config": {
            "leverage": env_config.leverage,
            "position_size_pct": env_config.position_size_pct,
            "stop_loss_pct": env_config.stop_loss_pct,
            "profit_take": 0.02,
            "stop_loss": 0.01,
            "horizon_hours": 24,
        },
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {torch_path}")
    print(f"  - {onnx_path}")
    print(f"  - {config_path}")
    print(f"\nInput tensor shape: (batch, {input_dim})")
    print(f"  - Candles: {candle_window} x {candle_features} = {candle_window * candle_features}")
    print(f"  - ML features: {ml_dim}")
    print(f"  - Portfolio: 2")
    print(f"\nOutput: action probabilities [short, hold, long]")


if __name__ == "__main__":
    main()
