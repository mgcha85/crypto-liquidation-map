"""
Liquidation Map Calculator

Calculates liquidation price levels based on Open Interest and leverage assumptions.

Algorithm:
1. Load OI and Price (OHLCV) data
2. Estimate position entry prices based on OI accumulation
3. Calculate liquidation prices for each leverage level
4. Build heatmap of liquidation levels
"""

from dataclasses import dataclass
from typing import Literal

import polars as pl

# Default leverage distribution assumption
DEFAULT_LEVERAGE_DISTRIBUTION = {
    10: 0.30,   # 30% of positions at 10x
    25: 0.35,   # 35% at 25x
    50: 0.25,   # 25% at 50x
    100: 0.10,  # 10% at 100x
}

# Binance maintenance margin rates (simplified)
MAINTENANCE_MARGIN = {
    10: 0.004,
    25: 0.005,
    50: 0.01,
    100: 0.025,
}


@dataclass
class LiquidationLevel:
    """A liquidation level on the map."""
    price: float
    volume: float  # Estimated liquidation volume at this price
    side: Literal["long", "short"]
    leverage: int


class LiquidationMapCalculator:
    """
    Calculate liquidation map from OI and price data.
    
    Example:
        >>> calc = LiquidationMapCalculator()
        >>> df_oi = pl.read_parquet("data/processed/open_interest.parquet")
        >>> df_price = pl.read_parquet("data/processed/klines.parquet")
        >>> heatmap = calc.calculate(df_oi, df_price)
    """
    
    def __init__(
        self,
        leverage_distribution: dict[int, float] | None = None,
        price_bucket_size: float = 100.0,  # $100 buckets for BTC
    ):
        self.leverage_distribution = leverage_distribution or DEFAULT_LEVERAGE_DISTRIBUTION
        self.price_bucket_size = price_bucket_size
    
    def calculate_liq_price_long(
        self,
        entry_price: float,
        leverage: int,
    ) -> float:
        """
        Calculate liquidation price for long position.
        
        Formula: Liq_long = EntryPrice × (1 - 1/Leverage + MaintenanceMargin)
        """
        mm = MAINTENANCE_MARGIN.get(leverage, 0.01)
        return entry_price * (1 - 1/leverage + mm)
    
    def calculate_liq_price_short(
        self,
        entry_price: float,
        leverage: int,
    ) -> float:
        """
        Calculate liquidation price for short position.
        
        Formula: Liq_short = EntryPrice × (1 + 1/Leverage - MaintenanceMargin)
        """
        mm = MAINTENANCE_MARGIN.get(leverage, 0.01)
        return entry_price * (1 + 1/leverage - mm)
    
    def estimate_position_entries(
        self,
        df_oi: pl.DataFrame,
        df_price: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Estimate position entries based on OI changes.
        
        When OI increases and price is moving up: likely new longs
        When OI increases and price is moving down: likely new shorts
        
        Returns:
            DataFrame with columns: timestamp, price, oi_delta, estimated_side
        """
        # TODO: Implement OI delta analysis
        # TODO: Correlate with price movement direction
        raise NotImplementedError("Coming soon")
    
    def build_liquidation_heatmap(
        self,
        df_entries: pl.DataFrame,
        current_price: float,
    ) -> pl.DataFrame:
        """
        Build liquidation heatmap from estimated entries.
        
        Returns:
            DataFrame with columns: price_bucket, long_volume, short_volume, total_volume
        """
        # TODO: Calculate liquidation prices for each entry
        # TODO: Aggregate into price buckets
        # TODO: Filter out already-liquidated positions
        raise NotImplementedError("Coming soon")
    
    def calculate(
        self,
        df_oi: pl.DataFrame,
        df_price: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Full pipeline: estimate entries → build heatmap.
        
        Returns:
            Liquidation heatmap DataFrame
        """
        df_entries = self.estimate_position_entries(df_oi, df_price)
        current_price = df_price.select(pl.last("close")).item()
        return self.build_liquidation_heatmap(df_entries, current_price)
