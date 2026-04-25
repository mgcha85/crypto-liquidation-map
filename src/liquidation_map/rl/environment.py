import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any

from .features import RLFeatureExtractor, RLFeatureConfig


@dataclass
class EnvConfig:
    initial_balance: float = 100_000.0
    position_size_pct: float = 0.25
    leverage: float = 2.0
    stop_loss_pct: float = 0.05
    commission_rate: float = 0.0004
    slippage_bps: float = 5.0
    max_position: int = 1
    liquidation_threshold: float = 0.9
    reward_scaling: float = 1.0
    trade_penalty: float = 0.0
    hold_bonus: float = 0.0
    inactivity_penalty: float = 0.001
    max_inactive_steps: int = 50


class CryptoFuturesEnv:
    
    def __init__(
        self,
        df_klines: pl.DataFrame,
        df_features: pl.DataFrame,
        config: Optional[EnvConfig] = None,
        feature_config: Optional[RLFeatureConfig] = None,
        mode: str = "train",
    ):
        self.config = config or EnvConfig()
        self.feature_config = feature_config or RLFeatureConfig()
        self.mode = mode
        
        self.df_klines = df_klines.sort("timestamp")
        self.df_features = df_features.sort("timestamp")
        
        self._align_data()
        
        self.feature_extractor = RLFeatureExtractor(self.feature_config)
        self.feature_extractor.fit(self.df_klines, self.df_features)
        
        self.action_dim = 3
        
        self._prices = self.df_klines["close"].to_numpy()
        self._timestamps = self.df_klines["timestamp"].to_list()
        self._labels = self.df_features["label"].to_numpy() if "label" in self.df_features.columns else None
        
        self.reset()
    
    def _align_data(self):
        kline_ts = set(self.df_klines["timestamp"].to_list())
        feature_ts = set(self.df_features["timestamp"].to_list())
        common_ts = sorted(kline_ts & feature_ts)
        
        if len(common_ts) < len(kline_ts):
            self.df_klines = self.df_klines.filter(pl.col("timestamp").is_in(common_ts))
            self.df_features = self.df_features.filter(pl.col("timestamp").is_in(common_ts))
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        if seed is not None:
            np.random.seed(seed)
        
        self.balance = self.config.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.position_value_at_entry = 0.0
        
        self.current_idx = self.feature_config.candle_window
        self.start_idx = self.current_idx
        
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = self.balance
        self.inactive_steps = 0
        
        self.trade_history = []
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        action_map = {0: -1, 1: 0, 2: 1}
        target_position = action_map[action]
        
        current_price = self._prices[self.current_idx]
        
        self._update_unrealized_pnl(current_price)
        
        sl_triggered = self._check_stop_loss(current_price)
        if sl_triggered:
            target_position = 0
        
        reward = self._execute_action(target_position, current_price)
        
        self.current_idx += 1
        
        terminated = self.current_idx >= len(self._prices) - 1
        truncated = self.balance <= 0
        
        equity = self.balance + self.unrealized_pnl
        if equity < self.config.initial_balance * (1 - self.config.liquidation_threshold):
            truncated = True
            reward -= 10.0
        
        if not terminated and not truncated:
            new_price = self._prices[self.current_idx]
            self._update_unrealized_pnl(new_price)
            
            pnl_reward = self._calculate_pnl_reward(current_price, new_price)
            reward += pnl_reward
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _check_stop_loss(self, current_price: float) -> bool:
        if self.position == 0 or self.config.stop_loss_pct is None:
            return False
        
        price_return = (current_price - self.entry_price) / self.entry_price
        position_return = self.position * price_return
        leveraged_return = position_return * self.config.leverage
        
        if leveraged_return <= -self.config.stop_loss_pct:
            return True
        return False
    
    def _execute_action(self, target_position: int, price: float) -> float:
        reward = 0.0
        
        if target_position == self.position:
            self.inactive_steps += 1
            if self.inactive_steps > self.config.max_inactive_steps and self.position == 0:
                reward -= self.config.inactivity_penalty
            return reward
        
        self.inactive_steps = 0
        
        if self.position != 0:
            exit_price = price * (1 - self.position * self.config.slippage_bps / 10000)
            price_return = (exit_price - self.entry_price) / self.entry_price
            pnl = self.position * price_return * self.position_value_at_entry
            commission = self.position_value_at_entry * self.config.commission_rate
            
            realized_pnl = pnl - commission
            self.balance += realized_pnl
            self.total_pnl += realized_pnl
            
            if realized_pnl > 0:
                self.winning_trades += 1
            
            self.trade_history.append({
                "exit_idx": self.current_idx,
                "exit_price": exit_price,
                "pnl": realized_pnl,
                "position": self.position,
            })
            
            self.position = 0
            self.entry_price = 0.0
            self.unrealized_pnl = 0.0
            self.position_value_at_entry = 0.0
        
        if target_position != 0:
            entry_price = price * (1 + target_position * self.config.slippage_bps / 10000)
            self.position_value_at_entry = self._position_value()
            commission = self.position_value_at_entry * self.config.commission_rate
            
            self.balance -= commission
            self.position = target_position
            self.entry_price = entry_price
            self.total_trades += 1
            
            self.trade_history.append({
                "entry_idx": self.current_idx,
                "entry_price": entry_price,
                "position": target_position,
            })
        
        reward -= self.config.trade_penalty
        
        self.peak_balance = max(self.peak_balance, self.balance + self.unrealized_pnl)
        
        return reward * self.config.reward_scaling
    
    def _calculate_pnl_reward(self, old_price: float, new_price: float) -> float:
        if self.position == 0:
            return 0.0
        
        price_change_pct = (new_price - old_price) / old_price
        position_return = self.position * price_change_pct * 100
        
        return position_return * self.config.reward_scaling
    
    def _update_unrealized_pnl(self, current_price: float):
        if self.position == 0 or self.position_value_at_entry == 0:
            self.unrealized_pnl = 0.0
        else:
            price_return = (current_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.position * price_return * self.position_value_at_entry
    
    def _position_value(self) -> float:
        return self.balance * self.config.position_size_pct * self.config.leverage
    
    def _get_observation(self) -> dict:
        return self.feature_extractor.extract_state(
            self.df_klines,
            self.df_features,
            self.current_idx,
            self.position,
            self.unrealized_pnl / self.config.initial_balance,
        )
    
    def _get_info(self) -> dict:
        total_value = self.balance + self.unrealized_pnl
        drawdown = (self.peak_balance - total_value) / self.peak_balance if self.peak_balance > 0 else 0
        
        return {
            "balance": self.balance,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "total_value": total_value,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "total_pnl": self.total_pnl,
            "drawdown": drawdown,
            "step": self.current_idx - self.start_idx,
            "timestamp": self._timestamps[self.current_idx] if self.current_idx < len(self._timestamps) else None,
        }
    
    def render(self, mode: str = "human"):
        info = self._get_info()
        print(f"Step {info['step']:4d} | "
              f"Balance: ${info['balance']:,.0f} | "
              f"Position: {info['position']:+d} | "
              f"PnL: ${info['total_pnl']:+,.0f} | "
              f"Trades: {info['total_trades']} | "
              f"WinRate: {info['win_rate']:.1%}")
    
    def get_metrics(self) -> dict:
        info = self._get_info()
        
        total_return = (info["total_value"] - self.config.initial_balance) / self.config.initial_balance
        
        if len(self.trade_history) > 0:
            pnls = [t.get("pnl", 0) for t in self.trade_history if "pnl" in t]
            if len(pnls) > 1:
                sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": info["drawdown"],
            "total_trades": info["total_trades"],
            "win_rate": info["win_rate"],
            "total_pnl": info["total_pnl"],
        }
