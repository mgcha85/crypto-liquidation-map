"""Tests for liquidation map calculator."""

import polars as pl
import pytest

from liquidation_map.analysis.liquidation_map import (
    LiquidationMapCalculator,
    MAINTENANCE_MARGIN,
    calculate_liq_price_long,
    calculate_liq_price_short,
)


class TestLiquidationPriceCalculation:
    
    def test_long_liquidation_10x(self):
        entry = 100_000
        liq_price = calculate_liq_price_long(entry, leverage=10)
        expected = entry * (1 - 0.1 + 0.004)
        assert abs(liq_price - expected) < 1
    
    def test_long_liquidation_25x(self):
        entry = 100_000
        liq_price = calculate_liq_price_long(entry, leverage=25)
        expected = entry * (1 - 0.04 + 0.005)
        assert abs(liq_price - expected) < 1
    
    def test_short_liquidation_10x(self):
        entry = 100_000
        liq_price = calculate_liq_price_short(entry, leverage=10)
        expected = entry * (1 + 0.1 - 0.004)
        assert abs(liq_price - expected) < 1
    
    def test_short_liquidation_25x(self):
        entry = 100_000
        liq_price = calculate_liq_price_short(entry, leverage=25)
        expected = entry * (1 + 0.04 - 0.005)
        assert abs(liq_price - expected) < 1


class TestMaintenanceMargin:
    
    def test_mm_increases_with_leverage(self):
        leverages = sorted(MAINTENANCE_MARGIN.keys())
        margins = [MAINTENANCE_MARGIN[lev] for lev in leverages]
        for i in range(1, len(margins)):
            assert margins[i] >= margins[i-1]


class TestLiquidationMapCalculator:
    
    @pytest.fixture
    def sample_data(self):
        df_oi = pl.DataFrame({
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 1, 1, 5),
                "1h",
                eager=True,
            ),
            "symbol": ["BTCUSDT"] * 6,
            "sum_open_interest": [1000.0, 1100.0, 1200.0, 1150.0, 1250.0, 1300.0],
            "sum_open_interest_value": [100_000_000.0, 110_000_000.0, 120_000_000.0, 
                                         115_000_000.0, 125_000_000.0, 130_000_000.0],
        })
        
        df_klines = pl.DataFrame({
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 1, 1, 5),
                "1h",
                eager=True,
            ),
            "symbol": ["BTCUSDT"] * 6,
            "open": [100000.0, 100500.0, 101000.0, 100800.0, 100200.0, 100500.0],
            "high": [100600.0, 101100.0, 101500.0, 101000.0, 100700.0, 101000.0],
            "low": [99800.0, 100400.0, 100900.0, 100100.0, 99900.0, 100200.0],
            "close": [100500.0, 101000.0, 100800.0, 100200.0, 100500.0, 100800.0],
            "volume": [1000.0] * 6,
        })
        
        return df_oi, df_klines
    
    def test_estimate_entries_returns_dataframe(self, sample_data):
        df_oi, df_klines = sample_data
        calc = LiquidationMapCalculator()
        
        entries = calc.estimate_position_entries(df_oi, df_klines)
        
        assert isinstance(entries, pl.DataFrame)
        assert "entry_price" in entries.columns
        assert "side" in entries.columns
        assert "oi_delta" in entries.columns
    
    def test_calculate_returns_buckets(self, sample_data):
        df_oi, df_klines = sample_data
        calc = LiquidationMapCalculator(price_bucket_size=1000)
        
        result = calc.calculate(df_oi, df_klines, current_price=100500)
        
        assert isinstance(result, pl.DataFrame)
        assert "price_bucket" in result.columns
        assert "long_volume" in result.columns
        assert "short_volume" in result.columns
    
    def test_empty_data_returns_empty(self):
        calc = LiquidationMapCalculator()
        
        df_oi = pl.DataFrame({
            "timestamp": [],
            "sum_open_interest_value": [],
        }).cast({"timestamp": pl.Datetime, "sum_open_interest_value": pl.Float64})
        
        df_klines = pl.DataFrame({
            "timestamp": [],
            "close": [],
        }).cast({"timestamp": pl.Datetime, "close": pl.Float64})
        
        result = calc.calculate(df_oi, df_klines, current_price=100000)
        
        assert result.is_empty()
