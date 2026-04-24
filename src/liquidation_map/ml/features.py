"""Feature extraction from liquidation map for ML training."""

from dataclasses import dataclass
import numpy as np
import polars as pl

from ..analysis.liquidation_map import LiquidationMapCalculator


@dataclass
class LiqMapFeatures:
    """
    Extracted features from a liquidation map window.
    
    Spatial features (from price distribution):
    - total_intensity: sum of all liquidation volume
    - above_below_ratio: volume above price / volume below price
    - near_price_concentration: volume within ±2% of current price
    - largest_cluster_distance_above/below: distance to largest cluster
    - entropy: concentration vs diffusion measure
    
    Temporal features (changes over window):
    - intensity_change_1h/6h/12h: rolling changes
    - cluster_shift: movement of largest clusters
    """
    timestamp: np.datetime64
    current_price: float
    
    total_intensity: float
    long_intensity: float
    short_intensity: float
    above_below_ratio: float
    
    near_1pct_concentration: float
    near_2pct_concentration: float
    near_5pct_concentration: float
    
    largest_long_cluster_distance: float
    largest_short_cluster_distance: float
    largest_long_cluster_volume: float
    largest_short_cluster_volume: float
    
    top3_long_distances: list[float]
    top3_short_distances: list[float]
    
    entropy: float
    skewness: float


class FeatureExtractor:
    
    def __init__(
        self,
        price_bucket_size: float = 100.0,
        num_price_bins: int = 128,
        price_range_pct: float = 0.15,
    ):
        self.price_bucket_size = price_bucket_size
        self.num_price_bins = num_price_bins
        self.price_range_pct = price_range_pct
        self.calculator = LiquidationMapCalculator(price_bucket_size=price_bucket_size)
    
    def extract_window_features(
        self,
        df_oi: pl.DataFrame,
        df_klines: pl.DataFrame,
        current_price: float,
    ) -> dict:
        df_map = self.calculator.calculate(df_oi, df_klines, current_price)
        
        if df_map.is_empty():
            return self._empty_features(current_price)
        
        long_vol = df_map["long_volume"].to_numpy()
        short_vol = df_map["short_volume"].to_numpy()
        prices = df_map["price_bucket"].to_numpy()
        total_vol = long_vol + short_vol
        
        total_intensity = float(total_vol.sum())
        long_intensity = float(long_vol.sum())
        short_intensity = float(short_vol.sum())
        
        above_mask = prices > current_price
        below_mask = prices < current_price
        vol_above = total_vol[above_mask].sum()
        vol_below = total_vol[below_mask].sum()
        above_below_ratio = vol_above / max(vol_below, 1e-10)
        
        price_distances_pct = np.abs(prices - current_price) / current_price
        near_1pct = total_vol[price_distances_pct <= 0.01].sum() / max(total_intensity, 1e-10)
        near_2pct = total_vol[price_distances_pct <= 0.02].sum() / max(total_intensity, 1e-10)
        near_5pct = total_vol[price_distances_pct <= 0.05].sum() / max(total_intensity, 1e-10)
        
        long_below = long_vol[below_mask]
        prices_below = prices[below_mask]
        if len(long_below) > 0 and long_below.sum() > 0:
            largest_long_idx = np.argmax(long_below)
            largest_long_distance = (current_price - prices_below[largest_long_idx]) / current_price
            largest_long_volume = float(long_below[largest_long_idx])
            
            top3_long_idx = np.argsort(long_below)[-3:][::-1]
            top3_long_distances = [(current_price - prices_below[i]) / current_price for i in top3_long_idx]
        else:
            largest_long_distance = 0.0
            largest_long_volume = 0.0
            top3_long_distances = [0.0, 0.0, 0.0]
        
        short_above = short_vol[above_mask]
        prices_above = prices[above_mask]
        if len(short_above) > 0 and short_above.sum() > 0:
            largest_short_idx = np.argmax(short_above)
            largest_short_distance = (prices_above[largest_short_idx] - current_price) / current_price
            largest_short_volume = float(short_above[largest_short_idx])
            
            top3_short_idx = np.argsort(short_above)[-3:][::-1]
            top3_short_distances = [(prices_above[i] - current_price) / current_price for i in top3_short_idx]
        else:
            largest_short_distance = 0.0
            largest_short_volume = 0.0
            top3_short_distances = [0.0, 0.0, 0.0]
        
        normalized = total_vol / max(total_intensity, 1e-10)
        entropy = float(-np.sum(normalized * np.log(normalized + 1e-10)))
        
        weighted_price = np.sum(prices * total_vol) / max(total_intensity, 1e-10)
        skewness = (weighted_price - current_price) / current_price
        
        return {
            "total_intensity": total_intensity,
            "long_intensity": long_intensity,
            "short_intensity": short_intensity,
            "long_short_ratio": long_intensity / max(short_intensity, 1e-10),
            "above_below_ratio": above_below_ratio,
            "near_1pct_concentration": near_1pct,
            "near_2pct_concentration": near_2pct,
            "near_5pct_concentration": near_5pct,
            "largest_long_cluster_distance": largest_long_distance,
            "largest_short_cluster_distance": largest_short_distance,
            "largest_long_cluster_volume": largest_long_volume,
            "largest_short_cluster_volume": largest_short_volume,
            "top3_long_dist_1": top3_long_distances[0] if len(top3_long_distances) > 0 else 0.0,
            "top3_long_dist_2": top3_long_distances[1] if len(top3_long_distances) > 1 else 0.0,
            "top3_long_dist_3": top3_long_distances[2] if len(top3_long_distances) > 2 else 0.0,
            "top3_short_dist_1": top3_short_distances[0] if len(top3_short_distances) > 0 else 0.0,
            "top3_short_dist_2": top3_short_distances[1] if len(top3_short_distances) > 1 else 0.0,
            "top3_short_dist_3": top3_short_distances[2] if len(top3_short_distances) > 2 else 0.0,
            "entropy": entropy,
            "skewness": skewness,
        }
    
    def extract_candle_features(self, df_klines: pl.DataFrame) -> dict:
        if df_klines.is_empty():
            return self._empty_candle_features()
        
        closes = df_klines["close"].to_numpy()
        highs = df_klines["high"].to_numpy()
        lows = df_klines["low"].to_numpy()
        opens = df_klines["open"].to_numpy()
        volumes = df_klines["volume"].to_numpy()
        
        returns = np.diff(closes) / closes[:-1]
        
        current_price = closes[-1]
        
        return {
            "return_1h": returns[-1] if len(returns) > 0 else 0.0,
            "return_6h": (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0.0,
            "return_12h": (closes[-1] / closes[-12] - 1) if len(closes) >= 12 else 0.0,
            "return_24h": (closes[-1] / closes[-24] - 1) if len(closes) >= 24 else 0.0,
            "volatility_6h": float(np.std(returns[-6:])) if len(returns) >= 6 else 0.0,
            "volatility_24h": float(np.std(returns[-24:])) if len(returns) >= 24 else 0.0,
            "atr_24h": float(np.mean(highs[-24:] - lows[-24:])) / current_price if len(highs) >= 24 else 0.0,
            "volume_ma_ratio": volumes[-1] / max(np.mean(volumes[-24:]), 1e-10) if len(volumes) >= 24 else 1.0,
            "wick_ratio_upper": (highs[-1] - max(opens[-1], closes[-1])) / max(highs[-1] - lows[-1], 1e-10),
            "wick_ratio_lower": (min(opens[-1], closes[-1]) - lows[-1]) / max(highs[-1] - lows[-1], 1e-10),
            "price_position": (closes[-1] - lows[-24:].min()) / max(highs[-24:].max() - lows[-24:].min(), 1e-10) if len(closes) >= 24 else 0.5,
        }
    
    def extract_2d_heatmap(
        self,
        df_oi: pl.DataFrame,
        df_klines: pl.DataFrame,
        window_hours: int = 50,
    ) -> np.ndarray:
        """
        Extract 2D heatmap for CNN input.
        Shape: (window_hours, num_price_bins)
        Values: normalized liquidation intensity at each (time, price_bin)
        """
        if df_klines.is_empty():
            return np.zeros((window_hours, self.num_price_bins))
        
        current_price = float(df_klines["close"].to_list()[-1])
        price_min = current_price * (1 - self.price_range_pct)
        price_max = current_price * (1 + self.price_range_pct)
        bin_edges = np.linspace(price_min, price_max, self.num_price_bins + 1)
        
        df_ts = self.calculator.calculate_timeseries(df_oi, df_klines, time_bucket="1h")
        
        if df_ts.is_empty():
            return np.zeros((window_hours, self.num_price_bins))
        
        timestamps = df_ts["timestamp"].unique().sort().to_list()[-window_hours:]
        heatmap = np.zeros((len(timestamps), self.num_price_bins))
        
        for i, ts in enumerate(timestamps):
            row_data = df_ts.filter(pl.col("timestamp") == ts)
            for row in row_data.iter_rows(named=True):
                price = row["price_bucket"]
                vol = row["total_volume"]
                bin_idx = np.searchsorted(bin_edges, price) - 1
                if 0 <= bin_idx < self.num_price_bins:
                    heatmap[i, bin_idx] += vol
        
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        
        if len(timestamps) < window_hours:
            padding = np.zeros((window_hours - len(timestamps), self.num_price_bins))
            heatmap = np.vstack([padding, heatmap])
        
        return heatmap
    
    def _empty_features(self, current_price: float) -> dict:
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
    
    def _empty_candle_features(self) -> dict:
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
