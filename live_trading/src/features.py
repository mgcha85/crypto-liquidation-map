"""Feature extraction matching the backtest pipeline exactly."""

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import polars as pl

from .config import FEATURE_COLUMNS


@dataclass
class FeatureWindow:
    timestamp: datetime
    current_price: float
    features: dict[str, float]
    
    def to_dataframe(self) -> pl.DataFrame:
        data = {"timestamp": [self.timestamp], "close": [self.current_price]}
        data.update({k: [v] for k, v in self.features.items()})
        return pl.DataFrame(data)


class LiveFeatureExtractor:
    
    def __init__(self, price_bucket_size: float = 100.0):
        self.price_bucket_size = price_bucket_size
        self.feature_names = FEATURE_COLUMNS.copy()
    
    def extract(
        self,
        df_oi: pl.DataFrame,
        df_klines: pl.DataFrame,
        current_price: float,
        timestamp: datetime,
    ) -> FeatureWindow:
        liq_features = self._extract_liquidation_features(df_oi, df_klines, current_price)
        candle_features = self._extract_candle_features(df_klines)
        
        all_features = {**liq_features, **candle_features}
        
        return FeatureWindow(
            timestamp=timestamp,
            current_price=current_price,
            features=all_features,
        )
    
    def _extract_liquidation_features(
        self,
        df_oi: pl.DataFrame,
        df_klines: pl.DataFrame,
        current_price: float,
    ) -> dict[str, float]:
        if df_oi.is_empty() or df_klines.is_empty():
            return self._empty_liquidation_features()
        
        df_map = self._build_liquidation_map(df_oi, df_klines, current_price)
        
        if df_map.is_empty():
            return self._empty_liquidation_features()
        
        long_vol = df_map["long_volume"].to_numpy()
        short_vol = df_map["short_volume"].to_numpy()
        prices = df_map["price_bucket"].to_numpy()
        total_vol = long_vol + short_vol
        
        total_intensity = float(total_vol.sum())
        long_intensity = float(long_vol.sum())
        short_intensity = float(short_vol.sum())
        long_short_ratio = long_intensity / max(short_intensity, 1e-10)
        
        above_mask = prices > current_price
        below_mask = prices < current_price
        vol_above = total_vol[above_mask].sum()
        vol_below = total_vol[below_mask].sum()
        above_below_ratio = vol_above / max(vol_below, 1e-10)
        
        price_dist_pct = np.abs(prices - current_price) / current_price
        near_1pct = total_vol[price_dist_pct <= 0.01].sum() / max(total_intensity, 1e-10)
        near_2pct = total_vol[price_dist_pct <= 0.02].sum() / max(total_intensity, 1e-10)
        near_5pct = total_vol[price_dist_pct <= 0.05].sum() / max(total_intensity, 1e-10)
        
        long_below = long_vol[below_mask]
        prices_below = prices[below_mask]
        if len(long_below) > 0 and long_below.sum() > 0:
            largest_long_idx = np.argmax(long_below)
            largest_long_dist = (current_price - prices_below[largest_long_idx]) / current_price
            largest_long_vol = float(long_below[largest_long_idx])
        else:
            largest_long_dist = 0.0
            largest_long_vol = 0.0
        
        short_above = short_vol[above_mask]
        prices_above = prices[above_mask]
        if len(short_above) > 0 and short_above.sum() > 0:
            largest_short_idx = np.argmax(short_above)
            largest_short_dist = (prices_above[largest_short_idx] - current_price) / current_price
            largest_short_vol = float(short_above[largest_short_idx])
        else:
            largest_short_dist = 0.0
            largest_short_vol = 0.0
        
        top3_long = self._get_top3_distances(long_below, prices_below, current_price, "long")
        top3_short = self._get_top3_distances(short_above, prices_above, current_price, "short")
        
        entropy = self._calculate_entropy(total_vol)
        skewness = self._calculate_skewness(total_vol, prices, current_price)
        
        return {
            "total_intensity": total_intensity,
            "long_intensity": long_intensity,
            "short_intensity": short_intensity,
            "long_short_ratio": long_short_ratio,
            "above_below_ratio": above_below_ratio,
            "near_1pct_concentration": near_1pct,
            "near_2pct_concentration": near_2pct,
            "near_5pct_concentration": near_5pct,
            "largest_long_cluster_distance": largest_long_dist,
            "largest_short_cluster_distance": largest_short_dist,
            "largest_long_cluster_volume": largest_long_vol,
            "largest_short_cluster_volume": largest_short_vol,
            "top3_long_dist_1": top3_long[0],
            "top3_long_dist_2": top3_long[1],
            "top3_long_dist_3": top3_long[2],
            "top3_short_dist_1": top3_short[0],
            "top3_short_dist_2": top3_short[1],
            "top3_short_dist_3": top3_short[2],
            "entropy": entropy,
            "skewness": skewness,
        }
    
    def _extract_candle_features(self, df_klines: pl.DataFrame) -> dict[str, float]:
        if df_klines.is_empty() or len(df_klines) < 24:
            return self._empty_candle_features()
        
        closes = df_klines["close"].to_numpy()
        highs = df_klines["high"].to_numpy()
        lows = df_klines["low"].to_numpy()
        volumes = df_klines["volume"].to_numpy()
        
        return_1h = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0.0
        return_6h = (closes[-1] / closes[-7] - 1) if len(closes) >= 7 else 0.0
        return_12h = (closes[-1] / closes[-13] - 1) if len(closes) >= 13 else 0.0
        return_24h = (closes[-1] / closes[-25] - 1) if len(closes) >= 25 else 0.0
        
        returns_6h = np.diff(closes[-7:]) / closes[-7:-1] if len(closes) >= 7 else np.array([0])
        returns_24h = np.diff(closes[-25:]) / closes[-25:-1] if len(closes) >= 25 else np.array([0])
        volatility_6h = float(np.std(returns_6h))
        volatility_24h = float(np.std(returns_24h))
        
        tr = np.maximum(highs[-24:] - lows[-24:], 
                       np.abs(highs[-24:] - np.roll(closes[-24:], 1)))
        tr = np.maximum(tr, np.abs(lows[-24:] - np.roll(closes[-24:], 1)))
        atr_24h = float(np.mean(tr[1:]))
        
        vol_ma = np.mean(volumes[-24:])
        volume_ma_ratio = volumes[-1] / max(vol_ma, 1e-10)
        
        body = closes[-1] - df_klines["open"].to_numpy()[-1]
        total_range = highs[-1] - lows[-1]
        if total_range > 0:
            wick_upper = (highs[-1] - max(closes[-1], df_klines["open"].to_numpy()[-1])) / total_range
            wick_lower = (min(closes[-1], df_klines["open"].to_numpy()[-1]) - lows[-1]) / total_range
        else:
            wick_upper = 0.0
            wick_lower = 0.0
        
        high_24 = np.max(highs[-24:])
        low_24 = np.min(lows[-24:])
        price_position = (closes[-1] - low_24) / max(high_24 - low_24, 1e-10)
        
        return {
            "return_1h": return_1h,
            "return_6h": return_6h,
            "return_12h": return_12h,
            "return_24h": return_24h,
            "volatility_6h": volatility_6h,
            "volatility_24h": volatility_24h,
            "atr_24h": atr_24h,
            "volume_ma_ratio": volume_ma_ratio,
            "wick_ratio_upper": wick_upper,
            "wick_ratio_lower": wick_lower,
            "price_position": price_position,
        }
    
    def _build_liquidation_map(
        self,
        df_oi: pl.DataFrame,
        df_klines: pl.DataFrame,
        current_price: float,
    ) -> pl.DataFrame:
        leverage_tiers = [5, 10, 20, 25, 50, 75, 100, 125]
        mm_rates = {5: 0.004, 10: 0.005, 20: 0.01, 25: 0.025, 50: 0.05, 75: 0.1, 100: 0.125, 125: 0.15}
        
        df_oi_sorted = df_oi.sort("timestamp")
        if "sumOpenInterest" not in df_oi.columns:
            return pl.DataFrame()
        
        oi_values = df_oi_sorted["sumOpenInterest"].to_numpy()
        timestamps = df_oi_sorted["timestamp"].to_numpy()
        
        df_klines_sorted = df_klines.sort("open_time")
        
        price_buckets = {}
        
        for i in range(1, len(oi_values)):
            oi_change = oi_values[i] - oi_values[i-1]
            if abs(oi_change) < 1e-10:
                continue
            
            ts = timestamps[i]
            entry_price = self._get_price_at_time(df_klines_sorted, ts)
            if entry_price is None:
                continue
            
            for leverage in leverage_tiers:
                mm = mm_rates[leverage]
                
                long_liq = entry_price * (1 - 1/leverage + mm)
                short_liq = entry_price * (1 + 1/leverage - mm)
                
                long_bucket = int(long_liq / self.price_bucket_size) * self.price_bucket_size
                short_bucket = int(short_liq / self.price_bucket_size) * self.price_bucket_size
                
                weight = abs(oi_change) / len(leverage_tiers)
                
                if oi_change > 0:
                    if long_bucket not in price_buckets:
                        price_buckets[long_bucket] = {"long": 0.0, "short": 0.0}
                    price_buckets[long_bucket]["long"] += weight * 0.5
                    
                    if short_bucket not in price_buckets:
                        price_buckets[short_bucket] = {"long": 0.0, "short": 0.0}
                    price_buckets[short_bucket]["short"] += weight * 0.5
        
        if not price_buckets:
            return pl.DataFrame()
        
        return pl.DataFrame({
            "price_bucket": list(price_buckets.keys()),
            "long_volume": [v["long"] for v in price_buckets.values()],
            "short_volume": [v["short"] for v in price_buckets.values()],
        })
    
    def _get_price_at_time(self, df_klines: pl.DataFrame, timestamp) -> float | None:
        if df_klines.is_empty():
            return None
        
        ts_ms = int(np.datetime64(timestamp, "ms").astype(np.int64)) if not isinstance(timestamp, (int, float)) else int(timestamp)
        
        df_before = df_klines.filter(pl.col("open_time") <= ts_ms)
        if df_before.is_empty():
            return float(df_klines["close"][0])
        
        return float(df_before["close"][-1])
    
    def _get_top3_distances(
        self,
        volumes: np.ndarray,
        prices: np.ndarray,
        current_price: float,
        side: str,
    ) -> list[float]:
        if len(volumes) == 0 or volumes.sum() == 0:
            return [0.0, 0.0, 0.0]
        
        sorted_idx = np.argsort(volumes)[::-1][:3]
        distances = []
        
        for idx in sorted_idx:
            if side == "long":
                dist = (current_price - prices[idx]) / current_price
            else:
                dist = (prices[idx] - current_price) / current_price
            distances.append(float(dist))
        
        while len(distances) < 3:
            distances.append(0.0)
        
        return distances
    
    def _calculate_entropy(self, volumes: np.ndarray) -> float:
        if volumes.sum() == 0:
            return 0.0
        
        probs = volumes / volumes.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-10)))
    
    def _calculate_skewness(
        self,
        volumes: np.ndarray,
        prices: np.ndarray,
        current_price: float,
    ) -> float:
        if volumes.sum() == 0:
            return 0.0
        
        weighted_mean = np.average(prices, weights=volumes)
        weighted_var = np.average((prices - weighted_mean)**2, weights=volumes)
        weighted_std = np.sqrt(weighted_var)
        
        if weighted_std == 0:
            return 0.0
        
        weighted_skew = np.average(((prices - weighted_mean) / weighted_std)**3, weights=volumes)
        return float(weighted_skew)
    
    def _empty_liquidation_features(self) -> dict[str, float]:
        return {
            "total_intensity": 0.0,
            "long_intensity": 0.0,
            "short_intensity": 0.0,
            "long_short_ratio": 1.0,
            "above_below_ratio": 1.0,
            "near_1pct_concentration": 0.0,
            "near_2pct_concentration": 0.0,
            "near_5pct_concentration": 0.0,
            "largest_long_cluster_distance": 0.0,
            "largest_short_cluster_distance": 0.0,
            "largest_long_cluster_volume": 0.0,
            "largest_short_cluster_volume": 0.0,
            "top3_long_dist_1": 0.0,
            "top3_long_dist_2": 0.0,
            "top3_long_dist_3": 0.0,
            "top3_short_dist_1": 0.0,
            "top3_short_dist_2": 0.0,
            "top3_short_dist_3": 0.0,
            "entropy": 0.0,
            "skewness": 0.0,
        }
    
    def _empty_candle_features(self) -> dict[str, float]:
        return {
            "return_1h": 0.0,
            "return_6h": 0.0,
            "return_12h": 0.0,
            "return_24h": 0.0,
            "volatility_6h": 0.0,
            "volatility_24h": 0.0,
            "atr_24h": 0.0,
            "volume_ma_ratio": 1.0,
            "wick_ratio_upper": 0.0,
            "wick_ratio_lower": 0.0,
            "price_position": 0.5,
        }
