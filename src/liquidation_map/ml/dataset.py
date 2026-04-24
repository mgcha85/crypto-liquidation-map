"""Training dataset generator with windowed features and labels."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

from .pipeline import DataPipeline
from .features import FeatureExtractor
from .labeling import TripleBarrierLabeler, BarrierConfig


@dataclass 
class WindowConfig:
    lookback_hours: int = 50
    horizon_hours: int = 24
    step_hours: int = 1
    min_oi_rows: int = 100


class TrainingDataGenerator:
    
    def __init__(
        self,
        pipeline: DataPipeline,
        window_config: WindowConfig | None = None,
        barrier_config: BarrierConfig | None = None,
        price_bucket_size: float = 100.0,
    ):
        self.pipeline = pipeline
        self.window_config = window_config or WindowConfig()
        self.barrier_config = barrier_config or BarrierConfig()
        
        self.feature_extractor = FeatureExtractor(price_bucket_size=price_bucket_size)
        self.labeler = TripleBarrierLabeler(self.barrier_config)
    
    def generate_dataset(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        kline_interval: str = "1h",
        oi_interval: str = "5m",
        output_path: Path | None = None,
    ) -> pl.DataFrame:
        df_klines = self.pipeline.read_silver("klines", symbol, kline_interval, start_date, end_date)
        df_oi = self.pipeline.read_silver("open_interest", symbol, oi_interval, start_date, end_date)
        
        if df_klines.is_empty() or df_oi.is_empty():
            print(f"No data for {symbol} between {start_date} and {end_date}")
            return pl.DataFrame()
        
        print(f"Loaded {len(df_klines)} klines, {len(df_oi)} OI rows")
        
        df_labels = self.labeler.compute_labels(df_klines, return_details=True)
        
        rows = []
        timestamps = df_klines["timestamp"].to_list()
        lookback = timedelta(hours=self.window_config.lookback_hours)
        
        valid_start_idx = self.window_config.lookback_hours
        valid_end_idx = len(timestamps) - self.window_config.horizon_hours
        
        for i in range(valid_start_idx, valid_end_idx, self.window_config.step_hours):
            ts = timestamps[i]
            window_start = ts - lookback
            
            klines_window = df_klines.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            oi_window = df_oi.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            
            if len(oi_window) < self.window_config.min_oi_rows:
                continue
            
            current_price = float(df_klines.filter(pl.col("timestamp") == ts)["close"][0])
            
            liq_features = self.feature_extractor.extract_window_features(
                oi_window, klines_window, current_price
            )
            candle_features = self.feature_extractor.extract_candle_features(klines_window)
            
            label_row = df_labels.filter(pl.col("timestamp") == ts)
            if label_row.is_empty():
                continue
            
            row = {
                "timestamp": ts,
                "symbol": symbol,
                "current_price": current_price,
                "label": int(label_row["label"][0]),
                "touch_time": int(label_row["touch_time"][0]),
                "mfe": float(label_row["mfe"][0]),
                "mae": float(label_row["mae"][0]),
                **liq_features,
                **candle_features,
            }
            rows.append(row)
            
            if len(rows) % 1000 == 0:
                print(f"  Processed {len(rows)} samples...")
        
        if not rows:
            return pl.DataFrame()
        
        df_train = pl.DataFrame(rows)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_train.write_parquet(output_path, compression="zstd")
            print(f"Saved {len(df_train)} samples to {output_path}")
        
        return df_train
    
    def generate_2d_heatmap_dataset(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        kline_interval: str = "1h",
        oi_interval: str = "5m",
        output_dir: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        """
        Generate dataset with 2D heatmaps for CNN training.
        
        Returns:
            heatmaps: (N, lookback_hours, num_price_bins) array
            labels: (N,) array of labels
            metadata: DataFrame with timestamps and other info
        """
        df_klines = self.pipeline.read_silver("klines", symbol, kline_interval, start_date, end_date)
        df_oi = self.pipeline.read_silver("open_interest", symbol, oi_interval, start_date, end_date)
        
        if df_klines.is_empty() or df_oi.is_empty():
            return np.array([]), np.array([]), pl.DataFrame()
        
        df_labels = self.labeler.compute_labels(df_klines, return_details=True)
        
        timestamps = df_klines["timestamp"].to_list()
        lookback = timedelta(hours=self.window_config.lookback_hours)
        
        valid_start_idx = self.window_config.lookback_hours
        valid_end_idx = len(timestamps) - self.window_config.horizon_hours
        
        heatmaps = []
        labels = []
        metadata_rows = []
        
        for i in range(valid_start_idx, valid_end_idx, self.window_config.step_hours):
            ts = timestamps[i]
            window_start = ts - lookback
            
            klines_window = df_klines.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            oi_window = df_oi.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            
            if len(oi_window) < self.window_config.min_oi_rows:
                continue
            
            heatmap = self.feature_extractor.extract_2d_heatmap(
                oi_window, klines_window, self.window_config.lookback_hours
            )
            
            label_row = df_labels.filter(pl.col("timestamp") == ts)
            if label_row.is_empty():
                continue
            
            heatmaps.append(heatmap)
            labels.append(int(label_row["label"][0]))
            
            current_price = float(df_klines.filter(pl.col("timestamp") == ts)["close"][0])
            metadata_rows.append({
                "timestamp": ts,
                "symbol": symbol,
                "current_price": current_price,
                "label": int(label_row["label"][0]),
            })
            
            if len(heatmaps) % 500 == 0:
                print(f"  Processed {len(heatmaps)} heatmaps...")
        
        heatmaps_arr = np.array(heatmaps)
        labels_arr = np.array(labels)
        metadata_df = pl.DataFrame(metadata_rows)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "heatmaps.npy", heatmaps_arr)
            np.save(output_dir / "labels.npy", labels_arr)
            metadata_df.write_parquet(output_dir / "metadata.parquet")
            print(f"Saved {len(heatmaps_arr)} heatmaps to {output_dir}")
        
        return heatmaps_arr, labels_arr, metadata_df
