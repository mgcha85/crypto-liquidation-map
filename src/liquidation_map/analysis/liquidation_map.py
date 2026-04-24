"""
Liquidation Map Calculator

Algorithm:
1. Track OI changes with price correlation
2. OI increase + price up = longs, OI increase + price down = shorts  
3. Calculate liquidation prices per leverage tier
4. Aggregate into price buckets for heatmap
"""

from dataclasses import dataclass
from typing import Literal

import polars as pl

LEVERAGE_TIERS = [10, 25, 50, 100]

DEFAULT_LEVERAGE_WEIGHTS = {
    10: 0.30,
    25: 0.35,
    50: 0.25,
    100: 0.10,
}

MAINTENANCE_MARGIN = {
    10: 0.004,
    25: 0.005,
    50: 0.01,
    100: 0.025,
}


@dataclass
class LiquidationLevel:
    price: float
    volume: float
    side: Literal["long", "short"]
    leverage: int


def calculate_liq_price_long(entry_price: float, leverage: int) -> float:
    """Long liquidation: Entry × (1 - 1/Leverage + MM)"""
    mm = MAINTENANCE_MARGIN.get(leverage, 0.01)
    return entry_price * (1 - 1/leverage + mm)


def calculate_liq_price_short(entry_price: float, leverage: int) -> float:
    """Short liquidation: Entry × (1 + 1/Leverage - MM)"""
    mm = MAINTENANCE_MARGIN.get(leverage, 0.01)
    return entry_price * (1 + 1/leverage - mm)


class LiquidationMapCalculator:
    
    def __init__(
        self,
        leverage_weights: dict[int, float] | None = None,
        price_bucket_size: float = 100.0,
    ):
        self.leverage_weights = leverage_weights or DEFAULT_LEVERAGE_WEIGHTS
        self.price_bucket_size = price_bucket_size
    
    def estimate_position_entries(
        self,
        df_oi: pl.DataFrame,
        df_klines: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Estimate entries from OI delta + price direction.
        OI up + price up = longs, OI up + price down = shorts.
        """
        df_oi = df_oi.sort("timestamp")
        df_klines = df_klines.sort("timestamp")
        
        oi_with_delta = df_oi.with_columns([
            (pl.col("sum_open_interest_value") - pl.col("sum_open_interest_value").shift(1)).alias("oi_delta"),
        ]).drop_nulls()
        
        klines_hourly = df_klines.select([
            pl.col("timestamp").dt.truncate("1h").alias("timestamp"),
            pl.col("close"),
            (pl.col("close") - pl.col("close").shift(1)).alias("price_delta"),
        ]).drop_nulls()
        
        merged = oi_with_delta.join_asof(
            klines_hourly,
            on="timestamp",
            strategy="nearest",
            tolerance="30m",
        )
        
        entries = (
            merged
            .filter((pl.col("oi_delta") > 0) & pl.col("close").is_not_null())
            .with_columns([
                pl.when(pl.col("price_delta") >= 0)
                  .then(pl.lit("long"))
                  .otherwise(pl.lit("short"))
                  .alias("side"),
                pl.col("close").alias("entry_price"),
            ])
            .select(["timestamp", "entry_price", "oi_delta", "side"])
        )
        
        return entries
    
    def build_liquidation_levels(self, df_entries: pl.DataFrame) -> pl.DataFrame:
        """Calculate liquidation prices for each entry across leverage tiers."""
        rows = []
        
        for row in df_entries.iter_rows(named=True):
            entry_price = row["entry_price"]
            oi_delta = row["oi_delta"]
            side = row["side"]
            
            for leverage, weight in self.leverage_weights.items():
                volume = oi_delta * weight
                
                if side == "long":
                    liq_price = calculate_liq_price_long(entry_price, leverage)
                else:
                    liq_price = calculate_liq_price_short(entry_price, leverage)
                
                rows.append({
                    "liq_price": liq_price,
                    "volume": volume,
                    "side": side,
                    "leverage": leverage,
                })
        
        return pl.DataFrame(rows)
    
    def aggregate_to_buckets(
        self,
        df_levels: pl.DataFrame,
        current_price: float,
    ) -> pl.DataFrame:
        """
        Aggregate to price buckets, filtering already-liquidated positions.
        Long liq > current = liquidated, Short liq < current = liquidated.
        """
        df_active = df_levels.filter(
            ((pl.col("side") == "long") & (pl.col("liq_price") < current_price)) |
            ((pl.col("side") == "short") & (pl.col("liq_price") > current_price))
        )
        
        df_bucketed = df_active.with_columns([
            ((pl.col("liq_price") / self.price_bucket_size).floor() * self.price_bucket_size).alias("price_bucket")
        ])
        
        long_buckets = (
            df_bucketed
            .filter(pl.col("side") == "long")
            .group_by("price_bucket")
            .agg(pl.col("volume").sum().alias("long_volume"))
        )
        
        short_buckets = (
            df_bucketed
            .filter(pl.col("side") == "short")
            .group_by("price_bucket")
            .agg(pl.col("volume").sum().alias("short_volume"))
        )
        
        result = (
            long_buckets
            .join(short_buckets, on="price_bucket", how="full", coalesce=True)
            .fill_null(0)
            .with_columns([
                (pl.col("long_volume") + pl.col("short_volume")).alias("total_volume"),
            ])
            .sort("price_bucket")
        )
        
        return result
    
    def calculate(
        self,
        df_oi: pl.DataFrame,
        df_klines: pl.DataFrame,
        current_price: float | None = None,
    ) -> pl.DataFrame:
        """Full pipeline: entries → levels → buckets."""
        price: float
        if current_price is None:
            price = float(df_klines.select(pl.col("close").last()).item())
        else:
            price = current_price
        
        df_entries = self.estimate_position_entries(df_oi, df_klines)
        
        if df_entries.is_empty():
            return pl.DataFrame({
                "price_bucket": [],
                "long_volume": [],
                "short_volume": [],
                "total_volume": [],
            })
        
        df_levels = self.build_liquidation_levels(df_entries)
        return self.aggregate_to_buckets(df_levels, price)
    
    def calculate_cumulative(
        self,
        df_buckets: pl.DataFrame,
        current_price: float,
    ) -> pl.DataFrame:
        """Cumulative volume from current price (longs down, shorts up)."""
        longs = (
            df_buckets
            .filter(pl.col("price_bucket") < current_price)
            .sort("price_bucket", descending=True)
            .with_columns([pl.col("long_volume").cum_sum().alias("long_cumulative")])
        )
        
        shorts = (
            df_buckets
            .filter(pl.col("price_bucket") > current_price)
            .sort("price_bucket")
            .with_columns([pl.col("short_volume").cum_sum().alias("short_cumulative")])
        )
        
        return pl.concat([longs, shorts]).sort("price_bucket")
