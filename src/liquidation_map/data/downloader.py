"""
Binance Vision Data Downloader

Downloads historical data from data.binance.vision:
- Open Interest (OI)
- Liquidation Snapshots  
- OHLCV (Klines)
"""

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import aiohttp

BASE_URL = "https://data.binance.vision/"

DataType = Literal["metrics", "liquidationSnapshot", "klines"]
TradingType = Literal["um", "cm"]
TimeFrame = Literal["daily", "monthly"]


def _parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD or YYYY-MM format."""
    if len(date_str) == 7:
        return datetime.strptime(date_str, "%Y-%m").date()
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _generate_daily_dates(start: date, end: date) -> list[str]:
    """Generate list of date strings between start and end (inclusive)."""
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def _generate_monthly_dates(start: date, end: date) -> list[str]:
    """Generate list of month strings between start and end (inclusive)."""
    months = []
    current = start.replace(day=1)
    end_month = end.replace(day=1)
    while current <= end_month:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


class BinanceDataDownloader:
    """
    Async downloader for Binance public data.
    
    Example:
        >>> downloader = BinanceDataDownloader(output_dir="data/raw")
        >>> files = await downloader.download_klines("BTCUSDT", "2024-01-01", "2024-01-31")
    """
    
    def __init__(
        self,
        output_dir: str | Path = "data/raw",
        trading_type: TradingType = "um",
        max_concurrent: int = 5,
        timeout: int = 30,
    ):
        self.output_dir = Path(output_dir)
        self.trading_type = trading_type
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
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
        
        URL Patterns:
        - Metrics (OI): data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-01.zip
        - Liquidation: data/futures/um/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-2024-01-01.zip
        - Klines: data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01-01.zip
        """
        base = f"data/futures/{self.trading_type}/{time_frame}/{data_type}/{symbol.upper()}"
        
        if data_type == "klines" and interval:
            filename = f"{symbol.upper()}-{interval}-{date_str}.zip"
            return f"{BASE_URL}{base}/{interval}/{filename}"
        else:
            filename = f"{symbol.upper()}-{data_type}-{date_str}.zip"
            return f"{BASE_URL}{base}/{filename}"
    
    def _build_output_path(
        self,
        data_type: DataType,
        symbol: str,
        date_str: str,
        interval: str | None = None,
    ) -> Path:
        """Build local output path for downloaded file."""
        if interval:
            filename = f"{symbol.upper()}-{data_type}-{interval}-{date_str}.zip"
        else:
            filename = f"{symbol.upper()}-{data_type}-{date_str}.zip"
        return self.output_dir / data_type / symbol.upper() / filename
    
    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        output_path: Path,
    ) -> Path | None:
        """Download a single file with rate limiting. Returns path if successful."""
        if output_path.exists():
            return output_path
        
        async with self._semaphore:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        content = await response.read()
                        output_path.write_bytes(content)
                        return output_path
                    elif response.status == 404:
                        return None
                    else:
                        print(f"Error {response.status}: {url}")
                        return None
            except asyncio.TimeoutError:
                print(f"Timeout: {url}")
                return None
            except Exception as e:
                print(f"Download failed: {url} - {e}")
                return None
    
    async def _download_batch(
        self,
        data_type: DataType,
        symbol: str,
        date_strings: list[str],
        time_frame: TimeFrame,
        interval: str | None = None,
    ) -> list[Path]:
        """Download a batch of files concurrently."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            tasks = []
            for date_str in date_strings:
                url = self._build_url(data_type, symbol, time_frame, date_str, interval)
                output_path = self._build_output_path(data_type, symbol, date_str, interval)
                tasks.append(self._download_file(session, url, output_path))
            
            results = await asyncio.gather(*tasks)
            return [p for p in results if p is not None]
    
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
            start_date: Start date (YYYY-MM-DD for daily, YYYY-MM for monthly)
            end_date: End date (YYYY-MM-DD for daily, YYYY-MM for monthly)
            interval: Data interval (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            time_frame: "daily" or "monthly"
        
        Returns:
            List of downloaded file paths
        """
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        
        if time_frame == "daily":
            date_strings = _generate_daily_dates(start, end)
        else:
            date_strings = _generate_monthly_dates(start, end)
        
        print(f"Downloading {len(date_strings)} metrics files for {symbol}...")
        return await self._download_batch("metrics", symbol, date_strings, time_frame)
    
    async def download_liquidation_snapshot(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        time_frame: TimeFrame = "daily",
    ) -> list[Path]:
        """
        Download Liquidation Snapshot data.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_date: Start date
            end_date: End date
            time_frame: "daily" or "monthly"
        
        Returns:
            List of downloaded file paths
        """
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        
        if time_frame == "daily":
            date_strings = _generate_daily_dates(start, end)
        else:
            date_strings = _generate_monthly_dates(start, end)
        
        print(f"Downloading {len(date_strings)} Liquidation Snapshot files for {symbol}...")
        return await self._download_batch("liquidationSnapshot", symbol, date_strings, time_frame)
    
    async def download_klines(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h",
        time_frame: TimeFrame = "daily",
    ) -> list[Path]:
        """
        Download OHLCV (Klines) data.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_date: Start date
            end_date: End date  
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d)
            time_frame: "daily" or "monthly"
        
        Returns:
            List of downloaded file paths
        """
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        
        if time_frame == "daily":
            date_strings = _generate_daily_dates(start, end)
        else:
            date_strings = _generate_monthly_dates(start, end)
        
        print(f"Downloading {len(date_strings)} Klines ({interval}) files for {symbol}...")
        return await self._download_batch("klines", symbol, date_strings, time_frame, interval)
    
    async def download_all(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        kline_interval: str = "1h",
        oi_interval: str = "5m",
    ) -> dict[str, list[Path]]:
        """
        Download all data types for a symbol.
        
        Returns:
            Dict with keys: 'klines', 'open_interest', 'liquidation'
        """
        klines, oi, liq = await asyncio.gather(
            self.download_klines(symbol, start_date, end_date, kline_interval),
            self.download_open_interest(symbol, start_date, end_date, oi_interval),
            self.download_liquidation_snapshot(symbol, start_date, end_date),
        )
        
        return {
            "klines": klines,
            "open_interest": oi,
            "liquidation": liq,
        }
