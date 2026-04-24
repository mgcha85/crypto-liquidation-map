"""
Binance Vision Data Downloader

Downloads historical data from data.binance.vision including:
- Open Interest (OI)
- Liquidation Snapshots
- OHLCV (Klines)
"""

import asyncio
from pathlib import Path
from typing import Literal

import aiohttp

BASE_URL = "https://data.binance.vision/"

DataType = Literal["openInterest", "liquidationSnapshot", "klines"]
TradingType = Literal["um", "cm"]  # USDT-M or COIN-M futures
TimeFrame = Literal["daily", "monthly"]


class BinanceDataDownloader:
    """
    Async downloader for Binance public data.
    
    Example:
        >>> downloader = BinanceDataDownloader(output_dir="data/raw")
        >>> await downloader.download_open_interest("BTCUSDT", "2024-01", "2024-12")
    """
    
    def __init__(
        self,
        output_dir: str | Path = "data/raw",
        trading_type: TradingType = "um",
        max_concurrent: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.trading_type = trading_type
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    def _build_url(
        self,
        data_type: DataType,
        symbol: str,
        time_frame: TimeFrame,
        date_str: str,
        interval: str | None = None,
    ) -> str:
        """
        Build download URL for Binance Vision.
        
        URL Pattern:
        - OI: data/futures/um/daily/openInterest/BTCUSDT/BTCUSDT-openInterest-5m-2024-01-01.zip
        - Liquidation: data/futures/um/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-2024-01-01.zip
        - Klines: data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-01.zip
        """
        base = f"data/futures/{self.trading_type}/{time_frame}/{data_type}/{symbol.upper()}"
        
        if data_type == "klines" and interval:
            filename = f"{symbol.upper()}-{interval}-{date_str}.zip"
            return f"{BASE_URL}{base}/{interval}/{filename}"
        elif data_type == "openInterest" and interval:
            filename = f"{symbol.upper()}-{data_type}-{interval}-{date_str}.zip"
            return f"{BASE_URL}{base}/{filename}"
        else:
            filename = f"{symbol.upper()}-{data_type}-{date_str}.zip"
            return f"{BASE_URL}{base}/{filename}"
    
    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        output_path: Path,
    ) -> bool:
        """Download a single file with rate limiting."""
        async with self._semaphore:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        content = await response.read()
                        output_path.write_bytes(content)
                        return True
                    elif response.status == 404:
                        # Data not available for this date
                        return False
                    else:
                        print(f"Error {response.status}: {url}")
                        return False
            except Exception as e:
                print(f"Download failed: {url} - {e}")
                return False
    
    async def download_open_interest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "5m",
        time_frame: TimeFrame = "daily",
    ) -> list[Path]:
        """
        Download Open Interest data.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_date: Start date (YYYY-MM-DD or YYYY-MM for monthly)
            end_date: End date (YYYY-MM-DD or YYYY-MM for monthly)
            interval: Data interval (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            time_frame: "daily" or "monthly"
        
        Returns:
            List of downloaded file paths
        """
        # TODO: Implement date range iteration
        # TODO: Implement concurrent downloads
        raise NotImplementedError("Coming soon")
    
    async def download_liquidation_snapshot(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        time_frame: TimeFrame = "daily",
    ) -> list[Path]:
        """Download Liquidation Snapshot data."""
        raise NotImplementedError("Coming soon")
    
    async def download_klines(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h",
        time_frame: TimeFrame = "daily",
    ) -> list[Path]:
        """Download OHLCV (Klines) data."""
        raise NotImplementedError("Coming soon")
