"""
LOCKED Configuration - DO NOT MODIFY without backtest validation.

These parameters were optimized via Optuna and validated through extensive backtesting.
Any modification requires re-running the benchmark pipeline.

SOTA Reference:
- Total Return: +2.82%
- Sharpe Ratio: 5.19
- Max Drawdown: 0.89%
- Win Rate: 58.3%
- Test Period: 2026-01-01 ~ 2026-04-22
"""

from dataclasses import dataclass
from typing import Literal


# ============================================================================
# LOCKED PARAMETERS - Validated via Optuna optimization
# ============================================================================

@dataclass(frozen=True)
class XGBParams:
    """XGBoost hyperparameters - LOCKED from Optuna Trial #24."""
    
    max_depth: int = 8
    learning_rate: float = 0.013721713883241381
    n_estimators: int = 175
    subsample: float = 0.6814260926873211
    colsample_bytree: float = 0.6683699504391535
    gamma: float = 1.1830353419914514
    reg_alpha: float = 2.3666885979478645
    reg_lambda: float = 1.1269397326544215
    
    # Fixed params
    objective: str = "multi:softmax"
    num_class: int = 3
    tree_method: str = "hist"
    device: str = "cpu"
    random_state: int = 42


@dataclass(frozen=True)
class BarrierConfig:
    """Triple Barrier parameters - LOCKED."""
    
    profit_take: float = 0.02   # 2% take profit
    stop_loss: float = 0.01    # 1% stop loss
    horizon_hours: int = 48    # 48 hours max hold


@dataclass(frozen=True)
class PositionConfig:
    """Position sizing - LOCKED."""
    
    position_size_pct: float = 0.10  # 10% of capital per trade
    taker_fee_pct: float = 0.0004    # 0.04% taker fee
    slippage_bps: float = 5.0        # 0.05% slippage assumption
    leverage: int = 1                 # No leverage for safety


# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

FEATURE_COLUMNS = [
    # Liquidation Map Features (20)
    "total_intensity", "long_intensity", "short_intensity", "long_short_ratio",
    "above_below_ratio", "near_1pct_concentration", "near_2pct_concentration", 
    "near_5pct_concentration", "largest_long_cluster_distance", "largest_short_cluster_distance",
    "largest_long_cluster_volume", "largest_short_cluster_volume",
    "top3_long_dist_1", "top3_long_dist_2", "top3_long_dist_3",
    "top3_short_dist_1", "top3_short_dist_2", "top3_short_dist_3",
    "entropy", "skewness",
    # Candle Features (11)
    "return_1h", "return_6h", "return_12h", "return_24h",
    "volatility_6h", "volatility_24h", "atr_24h", "volume_ma_ratio",
    "wick_ratio_upper", "wick_ratio_lower", "price_position",
]


# ============================================================================
# RUNTIME CONFIGURATION
# ============================================================================

@dataclass
class TradingConfig:
    """Runtime trading configuration."""
    
    # Mode
    mode: Literal["paper", "live"] = "paper"
    
    # Symbol
    symbol: str = "BTCUSDT"
    
    # Data windows
    lookback_hours: int = 50      # Hours of data for feature extraction
    update_interval_sec: int = 3600  # 1 hour update cycle
    
    # Risk limits
    daily_loss_limit_pct: float = 0.02   # -2% daily loss → halt
    weekly_loss_limit_pct: float = 0.05  # -5% weekly loss → full stop
    max_positions: int = 1               # Max 1 position at a time
    
    # API
    api_timeout_sec: int = 30
    max_retries: int = 3
    retry_delay_sec: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_trades: bool = True
    log_features: bool = False  # Set True for debugging parity
    
    # Paths
    model_path: str = "models/xgb_optuna_best.json"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_XGB_PARAMS = XGBParams()
DEFAULT_BARRIER = BarrierConfig()
DEFAULT_POSITION = PositionConfig()
DEFAULT_TRADING = TradingConfig()


def get_xgb_params_dict() -> dict:
    """Return XGBoost params as dict for model initialization."""
    p = DEFAULT_XGB_PARAMS
    return {
        "objective": p.objective,
        "num_class": p.num_class,
        "max_depth": p.max_depth,
        "learning_rate": p.learning_rate,
        "n_estimators": p.n_estimators,
        "subsample": p.subsample,
        "colsample_bytree": p.colsample_bytree,
        "gamma": p.gamma,
        "reg_alpha": p.reg_alpha,
        "reg_lambda": p.reg_lambda,
        "tree_method": p.tree_method,
        "device": p.device,
        "random_state": p.random_state,
    }
