"""Triple-barrier labeling for ML training."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl


@dataclass
class BarrierConfig:
    """
    Triple-barrier labeling configuration.
    
    profit_take: % gain to trigger +1 label
    stop_loss: % loss to trigger -1 label  
    horizon: max bars to wait before timeout (0 label or return sign)
    """
    profit_take: float = 0.02
    stop_loss: float = 0.01
    horizon: int = 24
    use_atr: bool = False
    atr_multiplier_pt: float = 2.0
    atr_multiplier_sl: float = 1.0


class TripleBarrierLabeler:
    
    def __init__(self, config: BarrierConfig | None = None):
        self.config = config or BarrierConfig()
    
    def compute_labels(
        self,
        df_klines: pl.DataFrame,
        return_details: bool = False,
    ) -> pl.DataFrame:
        """
        Compute triple-barrier labels for each timestamp.
        
        Labels:
            +1: price hit profit_take first
            -1: price hit stop_loss first
             0: neither hit within horizon (or sign of return at horizon)
        """
        closes = df_klines["close"].to_numpy()
        highs = df_klines["high"].to_numpy()
        lows = df_klines["low"].to_numpy()
        timestamps = df_klines["timestamp"].to_list()
        
        n = len(closes)
        labels = np.zeros(n, dtype=np.int8)
        touch_times = np.zeros(n, dtype=np.int32)
        max_favorable = np.zeros(n, dtype=np.float64)
        max_adverse = np.zeros(n, dtype=np.float64)
        
        for i in range(n - self.config.horizon):
            entry_price = closes[i]
            
            if self.config.use_atr:
                atr = self._compute_atr(highs, lows, closes, i, 24)
                pt_price = entry_price * (1 + self.config.atr_multiplier_pt * atr / entry_price)
                sl_price = entry_price * (1 - self.config.atr_multiplier_sl * atr / entry_price)
            else:
                pt_price = entry_price * (1 + self.config.profit_take)
                sl_price = entry_price * (1 - self.config.stop_loss)
            
            pt_hit = -1
            sl_hit = -1
            mfe = 0.0
            mae = 0.0
            
            for j in range(1, self.config.horizon + 1):
                if i + j >= n:
                    break
                
                high_j = highs[i + j]
                low_j = lows[i + j]
                
                mfe = max(mfe, (high_j - entry_price) / entry_price)
                mae = max(mae, (entry_price - low_j) / entry_price)
                
                if pt_hit < 0 and high_j >= pt_price:
                    pt_hit = j
                if sl_hit < 0 and low_j <= sl_price:
                    sl_hit = j
                
                if pt_hit > 0 and sl_hit > 0:
                    break
            
            max_favorable[i] = mfe
            max_adverse[i] = mae
            
            if pt_hit > 0 and sl_hit > 0:
                if pt_hit < sl_hit:
                    labels[i] = 1
                    touch_times[i] = pt_hit
                elif sl_hit < pt_hit:
                    labels[i] = -1
                    touch_times[i] = sl_hit
                else:
                    final_return = (closes[min(i + self.config.horizon, n - 1)] - entry_price) / entry_price
                    labels[i] = 1 if final_return > 0 else -1
                    touch_times[i] = pt_hit
            elif pt_hit > 0:
                labels[i] = 1
                touch_times[i] = pt_hit
            elif sl_hit > 0:
                labels[i] = -1
                touch_times[i] = sl_hit
            else:
                final_return = (closes[min(i + self.config.horizon, n - 1)] - entry_price) / entry_price
                labels[i] = 1 if final_return > 0 else (-1 if final_return < 0 else 0)
                touch_times[i] = self.config.horizon
        
        result = df_klines.with_columns([
            pl.Series("label", labels),
            pl.Series("touch_time", touch_times),
            pl.Series("mfe", max_favorable),
            pl.Series("mae", max_adverse),
        ])
        
        if not return_details:
            result = result.select(["timestamp", "label"])
        
        return result
    
    def _compute_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        idx: int,
        period: int,
    ) -> float:
        start = max(0, idx - period)
        if start >= idx:
            return highs[idx] - lows[idx]
        
        tr_values = []
        for i in range(start, idx):
            high_low = highs[i] - lows[i]
            if i > 0:
                high_close = abs(highs[i] - closes[i - 1])
                low_close = abs(lows[i] - closes[i - 1])
                tr = max(high_low, high_close, low_close)
            else:
                tr = high_low
            tr_values.append(tr)
        
        return float(np.mean(tr_values)) if tr_values else highs[idx] - lows[idx]
    
    def compute_regression_labels(
        self,
        df_klines: pl.DataFrame,
        horizon: int = 24,
    ) -> pl.DataFrame:
        closes = df_klines["close"].to_numpy()
        n = len(closes)
        
        forward_returns = np.zeros(n, dtype=np.float64)
        for i in range(n - horizon):
            forward_returns[i] = (closes[i + horizon] - closes[i]) / closes[i]
        
        return df_klines.with_columns([
            pl.Series("forward_return", forward_returns),
        ])
