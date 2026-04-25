import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import Optional


@dataclass
class RLFeatureConfig:
    candle_window: int = 200
    use_liquidation_features: bool = True
    use_technical_features: bool = True
    normalize: bool = True


class RLFeatureExtractor:
    
    def __init__(self, config: Optional[RLFeatureConfig] = None):
        self.config = config or RLFeatureConfig()
        self._stats: dict = {}
    
    def fit(self, df_klines: pl.DataFrame, df_features: Optional[pl.DataFrame] = None):
        if not self.config.normalize:
            return self
        
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            if col in df_klines.columns:
                values = df_klines[col].to_numpy()
                self._stats[col] = {
                    "mean": np.nanmean(values),
                    "std": np.nanstd(values) + 1e-8,
                }
        
        if df_features is not None:
            for col in df_features.columns:
                if col not in ["timestamp", "label"]:
                    values = df_features[col].to_numpy()
                    self._stats[col] = {
                        "mean": np.nanmean(values),
                        "std": np.nanstd(values) + 1e-8,
                    }
        
        return self
    
    def extract_candle_window(
        self,
        df_klines: pl.DataFrame,
        end_idx: int,
    ) -> np.ndarray:
        start_idx = max(0, end_idx - self.config.candle_window)
        window = df_klines.slice(start_idx, end_idx - start_idx)
        
        ohlcv = np.zeros((self.config.candle_window, 5), dtype=np.float32)
        
        cols = ["open", "high", "low", "close", "volume"]
        actual_len = min(window.height, self.config.candle_window)
        offset = self.config.candle_window - actual_len
        
        for i, col in enumerate(cols):
            if col in window.columns:
                values = window[col].to_numpy()[-actual_len:]
                
                if self.config.normalize and col in self._stats:
                    values = (values - self._stats[col]["mean"]) / self._stats[col]["std"]
                
                ohlcv[offset:, i] = values
        
        return ohlcv
    
    def extract_ml_features(
        self,
        df_features: pl.DataFrame,
        idx: int,
    ) -> np.ndarray:
        if idx < 0 or idx >= df_features.height:
            return np.zeros(len(df_features.columns) - 2, dtype=np.float32)
        
        row = df_features.row(idx, named=True)
        
        feature_cols = [c for c in df_features.columns if c not in ["timestamp", "label"]]
        features = np.array([row.get(c, 0.0) for c in feature_cols], dtype=np.float32)
        
        if self.config.normalize:
            for i, col in enumerate(feature_cols):
                if col in self._stats:
                    features[i] = (features[i] - self._stats[col]["mean"]) / self._stats[col]["std"]
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def extract_state(
        self,
        df_klines: pl.DataFrame,
        df_features: pl.DataFrame,
        idx: int,
        position: int,
        unrealized_pnl: float,
    ) -> dict:
        candles = self.extract_candle_window(df_klines, idx)
        ml_features = self.extract_ml_features(df_features, idx)
        
        portfolio_state = np.array([
            position,
            unrealized_pnl,
        ], dtype=np.float32)
        
        return {
            "candles": candles,
            "ml_features": ml_features,
            "portfolio": portfolio_state,
        }
    
    @property
    def candle_shape(self) -> tuple:
        return (self.config.candle_window, 5)
    
    def ml_feature_dim(self, df_features: pl.DataFrame) -> int:
        return len([c for c in df_features.columns if c not in ["timestamp", "label"]])
