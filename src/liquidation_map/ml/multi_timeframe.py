"""Multi-timeframe strategy: loads 1m candles and resamples to 5m/15m/1h for comparison."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from .features import FeatureExtractor
from .labeling import TripleBarrierLabeler, BarrierConfig
from .backtest import Backtester, BacktestConfig, BacktestResult
from .models.xgboost_model import XGBoostModel, XGBConfig


CRYPTO_DATA_PATH = Path("/mnt/data/finance/cryptocurrency")

TIMEFRAME_CONFIGS = {
    "5m": {
        "interval_minutes": 5,
        "horizon_bars": 288,
        "lookback_bars": 600,
        "step_bars": 12,
        "profit_take": 0.02,
        "stop_loss": 0.01,
    },
    "15m": {
        "interval_minutes": 15,
        "horizon_bars": 96,
        "lookback_bars": 200,
        "step_bars": 4,
        "profit_take": 0.02,
        "stop_loss": 0.01,
    },
    "1h": {
        "interval_minutes": 60,
        "horizon_bars": 24,
        "lookback_bars": 50,
        "step_bars": 1,
        "profit_take": 0.02,
        "stop_loss": 0.01,
    },
}

PRICE_BUCKETS = {
    "BTCUSDT": 250.0,
    "ETHUSDT": 25.0,
    "SOLUSDT": 0.5,
}


@dataclass
class TimeframeResult:
    """Results for a single timeframe strategy."""
    timeframe: str
    symbol: str
    backtest: BacktestResult
    train_accuracy: float
    test_accuracy: float
    feature_importance: pl.DataFrame
    label_distribution: dict
    config: dict


@dataclass
class MultiTimeframeResults:
    """Aggregated results across timeframes."""
    symbol: str
    results: dict[str, TimeframeResult] = field(default_factory=dict)
    
    def to_comparison_df(self) -> pl.DataFrame:
        """Create comparison DataFrame for all timeframes."""
        rows = []
        for tf, result in self.results.items():
            rows.append({
                "timeframe": tf,
                "total_return": result.backtest.total_return,
                "sharpe_ratio": result.backtest.sharpe_ratio,
                "max_drawdown": result.backtest.max_drawdown,
                "win_rate": result.backtest.win_rate,
                "profit_factor": result.backtest.profit_factor,
                "num_trades": result.backtest.num_trades,
                "train_accuracy": result.train_accuracy,
                "test_accuracy": result.test_accuracy,
            })
        return pl.DataFrame(rows)


class MultiTimeframeLoader:
    """Load and resample 1-minute candles to target timeframes."""
    
    def __init__(self, data_path: Path = CRYPTO_DATA_PATH):
        self.data_path = data_path
    
    def load_1m_candles(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """Load 1-minute candles from Hive-partitioned parquet files."""
        symbol_path = self.data_path / symbol
        if not symbol_path.exists():
            raise FileNotFoundError(f"Symbol directory not found: {symbol_path}")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dfs = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            date_path = symbol_path / f"date={date_str}" / "00000000.parquet"
            
            if date_path.exists():
                df = pl.read_parquet(date_path)
                dfs.append(df)
            
            current += timedelta(days=1)
        
        if not dfs:
            return pl.DataFrame()
        
        df = pl.concat(dfs)
        
        if "datetime" not in df.columns and "open_time" in df.columns:
            df = df.with_columns([
                (pl.col("open_time") * 1_000_000).cast(pl.Datetime("us")).alias("datetime")
            ])
        
        return df.sort("datetime")
    
    def resample_to_timeframe(
        self,
        df_1m: pl.DataFrame,
        interval_minutes: int,
    ) -> pl.DataFrame:
        """Resample 1-minute candles to target timeframe."""
        if df_1m.is_empty():
            return pl.DataFrame()
        
        df_resampled = df_1m.group_by_dynamic(
            "datetime",
            every=f"{interval_minutes}m",
        ).agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume") if "quote_volume" in df_1m.columns else pl.lit(0.0).alias("quote_volume"),
            pl.col("trades").sum().alias("trades") if "trades" in df_1m.columns else pl.lit(0).alias("trades"),
        ])
        
        df_resampled = df_resampled.rename({"datetime": "timestamp"})
        
        return df_resampled
    
    def get_candles(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str,
    ) -> pl.DataFrame:
        """Load and resample candles to target timeframe."""
        config = TIMEFRAME_CONFIGS.get(timeframe)
        if not config:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        df_1m = self.load_1m_candles(symbol, start_date, end_date)
        if df_1m.is_empty():
            return pl.DataFrame()
        
        return self.resample_to_timeframe(df_1m, config["interval_minutes"])


class MultiTimeframeStrategy:
    """Run and compare strategies across multiple timeframes."""
    
    def __init__(
        self,
        symbol: str,
        oi_data_path: Path | str = "data/silver",
        train_cutoff: str = "2025-12-31",
    ):
        self.symbol = symbol
        self.oi_data_path = Path(oi_data_path)
        self.train_cutoff = train_cutoff
        self.loader = MultiTimeframeLoader()
        self.price_bucket = PRICE_BUCKETS.get(symbol, 100.0)
    
    def load_oi_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """Load Open Interest data from silver layer."""
        from .pipeline import DataPipeline
        pipeline = DataPipeline(silver_dir=self.oi_data_path)
        df_oi = pipeline.read_silver("open_interest", self.symbol, "5m", start_date, end_date)
        pipeline.close()
        return df_oi
    
    def generate_features(
        self,
        df_klines: pl.DataFrame,
        df_oi: pl.DataFrame,
        timeframe: str,
    ) -> pl.DataFrame:
        """Generate features for a given timeframe."""
        config = TIMEFRAME_CONFIGS[timeframe]
        
        barrier_config = BarrierConfig(
            profit_take=config["profit_take"],
            stop_loss=config["stop_loss"],
            horizon=config["horizon_bars"],
        )
        
        feature_extractor = FeatureExtractor(price_bucket_size=self.price_bucket)
        labeler = TripleBarrierLabeler(barrier_config)

        df_labels = labeler.compute_labels(df_klines, return_details=True)
        
        timestamps = df_klines["timestamp"].to_list()
        lookback_bars = config["lookback_bars"]
        horizon_bars = config["horizon_bars"]
        step_bars = config["step_bars"]
        interval_minutes = config["interval_minutes"]
        
        lookback_td = timedelta(minutes=lookback_bars * interval_minutes)
        
        rows = []
        valid_start_idx = lookback_bars
        valid_end_idx = len(timestamps) - horizon_bars
        
        for i in range(valid_start_idx, valid_end_idx, step_bars):
            ts = timestamps[i]
            window_start = ts - lookback_td
            
            klines_window = df_klines.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            oi_window = df_oi.filter(
                (pl.col("timestamp") >= window_start) & (pl.col("timestamp") < ts)
            )
            
            if len(oi_window) < 50:
                continue
            
            current_price = float(df_klines.filter(pl.col("timestamp") == ts)["close"][0])

            liq_features = feature_extractor.extract_window_features(
                oi_window, klines_window, current_price
            )
            candle_features = self._extract_candle_features_scaled(
                klines_window, interval_minutes
            )
            
            label_row = df_labels.filter(pl.col("timestamp") == ts)
            if label_row.is_empty():
                continue
            
            row = {
                "timestamp": ts,
                "symbol": self.symbol,
                "current_price": current_price,
                "label": int(label_row["label"][0]),
                **liq_features,
                **candle_features,
            }
            rows.append(row)
            
            if len(rows) % 1000 == 0:
                print(f"  [{timeframe}] Processed {len(rows)} samples...")
        
        if not rows:
            return pl.DataFrame()
        
        return pl.DataFrame(rows)
    
    def _extract_candle_features_scaled(
        self,
        df_klines: pl.DataFrame,
        interval_minutes: int,
    ) -> dict:
        """Extract candle features scaled for timeframe."""
        if df_klines.is_empty():
            return self._empty_candle_features()
        
        closes = df_klines["close"].to_numpy()
        highs = df_klines["high"].to_numpy()
        lows = df_klines["low"].to_numpy()
        opens = df_klines["open"].to_numpy()
        volumes = df_klines["volume"].to_numpy()
        
        returns = np.diff(closes) / closes[:-1]
        current_price = closes[-1]

        bars_per_hour = 60 // interval_minutes
        
        bars_1h = bars_per_hour
        bars_6h = 6 * bars_per_hour
        bars_12h = 12 * bars_per_hour
        bars_24h = 24 * bars_per_hour
        
        return {
            "return_1h": (closes[-1] / closes[-bars_1h] - 1) if len(closes) >= bars_1h else 0.0,
            "return_6h": (closes[-1] / closes[-bars_6h] - 1) if len(closes) >= bars_6h else 0.0,
            "return_12h": (closes[-1] / closes[-bars_12h] - 1) if len(closes) >= bars_12h else 0.0,
            "return_24h": (closes[-1] / closes[-bars_24h] - 1) if len(closes) >= bars_24h else 0.0,
            "volatility_6h": float(np.std(returns[-bars_6h:])) if len(returns) >= bars_6h else 0.0,
            "volatility_24h": float(np.std(returns[-bars_24h:])) if len(returns) >= bars_24h else 0.0,
            "atr_24h": float(np.mean(highs[-bars_24h:] - lows[-bars_24h:])) / current_price if len(highs) >= bars_24h else 0.0,
            "volume_ma_ratio": volumes[-1] / max(np.mean(volumes[-bars_24h:]), 1e-10) if len(volumes) >= bars_24h else 1.0,
            "wick_ratio_upper": (highs[-1] - max(opens[-1], closes[-1])) / max(highs[-1] - lows[-1], 1e-10),
            "wick_ratio_lower": (min(opens[-1], closes[-1]) - lows[-1]) / max(highs[-1] - lows[-1], 1e-10),
            "price_position": (closes[-1] - lows[-bars_24h:].min()) / max(highs[-bars_24h:].max() - lows[-bars_24h:].min(), 1e-10) if len(closes) >= bars_24h else 0.5,
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
    
    def run_single_timeframe(
        self,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> TimeframeResult | None:
        """Run strategy for a single timeframe."""
        print(f"\n{'='*60}")
        print(f"Running {self.symbol} - {timeframe} timeframe")
        print(f"{'='*60}")
        
        print(f"Loading {timeframe} candles...")
        df_klines = self.loader.get_candles(self.symbol, start_date, end_date, timeframe)
        if df_klines.is_empty():
            print(f"No candle data for {self.symbol}")
            return None
        print(f"  Loaded {len(df_klines)} candles")
        
        print("Loading OI data...")
        df_oi = self.load_oi_data(start_date, end_date)
        if df_oi.is_empty():
            print(f"No OI data for {self.symbol}")
            return None
        print(f"  Loaded {len(df_oi)} OI rows")

        print("Generating features...")
        df_features = self.generate_features(df_klines, df_oi, timeframe)
        if df_features.is_empty():
            print("No features generated")
            return None
        print(f"  Generated {len(df_features)} samples")

        cutoff = datetime.strptime(self.train_cutoff, "%Y-%m-%d")
        df_train = df_features.filter(pl.col("timestamp") < cutoff)
        df_test = df_features.filter(pl.col("timestamp") >= cutoff)
        
        print(f"  Train: {len(df_train)}, Test: {len(df_test)}")
        
        if len(df_train) < 100 or len(df_test) < 50:
            print("Insufficient data for training/testing")
            return None

        label_dist = df_train["label"].value_counts().to_dicts()
        label_dict = {row["label"]: row["count"] for row in label_dist}

        total = sum(label_dict.values())
        sell_count = label_dict.get(-1, 0)
        buy_count = label_dict.get(1, 0)
        scale_pos_weight = sell_count / max(buy_count, 1)
        
        print(f"  Label distribution: {label_dict}")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")

        xgb_config = XGBConfig(
            objective="binary:logistic",
            num_class=2,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )

        df_train_filtered = df_train.filter(pl.col("label") != 0).with_columns([
            pl.when(pl.col("label") == -1).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("label")
        ])
        df_test_filtered = df_test.filter(pl.col("label") != 0).with_columns([
            pl.when(pl.col("label") == -1).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("label")
        ])

        print(f"  Binary train: {len(df_train_filtered)}, test: {len(df_test_filtered)}")

        model = XGBoostModel(xgb_config)
        train_result = model.train(df_train_filtered)

        test_preds = model.predict(df_test_filtered)
        test_proba = model.predict_proba(df_test_filtered)

        test_labels = df_test_filtered["label"].to_numpy()
        test_acc = (test_preds == test_labels).mean()
        
        print(f"  Train accuracy: {train_result['train_accuracy']:.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")

        backtest_config = BacktestConfig(
            position_size_pct=0.1,
            taker_fee_pct=0.0004,
            slippage_bps=5.0,
        )
        backtester = Backtester(backtest_config)

        signals = np.where(test_preds == 0, 1, -1)
        
        backtest_result = backtester.run(df_test_filtered, signals, test_proba)
        
        print(f"\nBacktest Results ({timeframe}):")
        print(f"  Return: {backtest_result.total_return*100:.2f}%")
        print(f"  Sharpe: {backtest_result.sharpe_ratio:.2f}")
        print(f"  Max DD: {backtest_result.max_drawdown*100:.2f}%")
        print(f"  Win Rate: {backtest_result.win_rate*100:.1f}%")
        print(f"  Trades: {backtest_result.num_trades}")

        feat_imp = model.get_feature_importance()
        
        return TimeframeResult(
            timeframe=timeframe,
            symbol=self.symbol,
            backtest=backtest_result,
            train_accuracy=train_result["train_accuracy"],
            test_accuracy=test_acc,
            feature_importance=feat_imp,
            label_distribution=label_dict,
            config=TIMEFRAME_CONFIGS[timeframe],
        )
    
    def run_all_timeframes(
        self,
        start_date: str = "2020-09-01",
        end_date: str = "2026-04-22",
        timeframes: list[str] | None = None,
    ) -> MultiTimeframeResults:
        """Run strategy across all timeframes."""
        timeframes = timeframes or ["5m", "15m", "1h"]
        
        results = MultiTimeframeResults(symbol=self.symbol)
        
        for tf in timeframes:
            result = self.run_single_timeframe(tf, start_date, end_date)
            if result:
                results.results[tf] = result
        
        return results


def run_multi_timeframe_comparison(
    symbol: str = "BTCUSDT",
    start_date: str = "2020-09-01",
    end_date: str = "2026-04-22",
    train_cutoff: str = "2025-12-31",
    output_dir: Path | str = "data/train",
) -> MultiTimeframeResults:
    output_dir = Path(output_dir)

    strategy = MultiTimeframeStrategy(
        symbol=symbol,
        train_cutoff=train_cutoff,
    )

    results = strategy.run_all_timeframes(start_date, end_date)

    if results.results:
        comparison_df = results.to_comparison_df()
        output_path = output_dir / f"{symbol.lower()}_timeframe_comparison.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.write_parquet(output_path)
        print(f"\nSaved comparison to {output_path}")

        import json
        json_data = {
            "symbol": symbol,
            "timeframes": {}
        }
        for tf, result in results.results.items():
            json_data["timeframes"][tf] = {
                "total_return": result.backtest.total_return,
                "sharpe_ratio": result.backtest.sharpe_ratio,
                "max_drawdown": result.backtest.max_drawdown,
                "win_rate": result.backtest.win_rate,
                "profit_factor": result.backtest.profit_factor,
                "num_trades": result.backtest.num_trades,
                "train_accuracy": result.train_accuracy,
                "test_accuracy": result.test_accuracy,
            }

        json_path = output_dir / f"{symbol.lower()}_timeframe_results.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved JSON to {json_path}")

    return results


if __name__ == "__main__":
    results = run_multi_timeframe_comparison("BTCUSDT")
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(results.to_comparison_df())
