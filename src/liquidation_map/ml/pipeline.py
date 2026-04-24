"""
Data pipeline with Hive partitioning for ML training.

Directory structure:
    data/silver/
        dataset=klines/symbol=BTCUSDT/interval=1h/year=2024/month=04/day=24/data.parquet
        dataset=open_interest/symbol=BTCUSDT/interval=1h/year=2024/month=04/day=24/data.parquet
        dataset=liq_map/symbol=BTCUSDT/interval=1h/year=2024/month=04/day=24/data.parquet
    data/train/
        symbol=BTCUSDT/year=2024/month=04/windows.parquet
"""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal
import zipfile
import io

import polars as pl
import duckdb

from ..data.downloader import BinanceDataDownloader


@dataclass
class PartitionKey:
    dataset: str
    symbol: str
    interval: str
    year: int
    month: int
    day: int
    
    @property
    def path(self) -> str:
        return (
            f"dataset={self.dataset}/symbol={self.symbol}/interval={self.interval}/"
            f"year={self.year}/month={self.month:02d}/day={self.day:02d}"
        )
    
    @classmethod
    def from_date(cls, d: date, dataset: str, symbol: str, interval: str) -> "PartitionKey":
        return cls(dataset, symbol, interval, d.year, d.month, d.day)


class MetadataLedger:
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        self._init_schema()
    
    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS partitions (
                dataset VARCHAR,
                symbol VARCHAR,
                interval VARCHAR,
                day DATE,
                status VARCHAR DEFAULT 'pending',
                row_count INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (dataset, symbol, interval, day)
            )
        """)
    
    def get_missing_days(
        self,
        dataset: str,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        existing = self.conn.execute("""
            SELECT day FROM partitions
            WHERE dataset = ? AND symbol = ? AND interval = ? 
            AND status = 'complete' AND day BETWEEN ? AND ?
        """, [dataset, symbol, interval, start_date, end_date]).fetchall()
        
        existing_dates = {row[0] for row in existing}
        all_dates = []
        current = start_date
        while current <= end_date:
            if current not in existing_dates:
                all_dates.append(current)
            current += timedelta(days=1)
        return all_dates
    
    def mark_complete(self, key: PartitionKey, row_count: int):
        self.conn.execute("""
            INSERT OR REPLACE INTO partitions (dataset, symbol, interval, day, status, row_count, updated_at)
            VALUES (?, ?, ?, ?, 'complete', ?, CURRENT_TIMESTAMP)
        """, [key.dataset, key.symbol, key.interval, date(key.year, key.month, key.day), row_count])
    
    def close(self):
        self.conn.close()


class DataPipeline:
    
    KLINE_COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore"
    ]
    
    OI_COLUMNS = [
        "create_time", "symbol", "sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"
    ]
    
    def __init__(
        self,
        raw_dir: Path | str = "data/raw",
        silver_dir: Path | str = "data/silver",
        train_dir: Path | str = "data/train",
        metadata_db: Path | str = "data/metadata.duckdb",
    ):
        self.raw_dir = Path(raw_dir)
        self.silver_dir = Path(silver_dir)
        self.train_dir = Path(train_dir)
        self.ledger = MetadataLedger(Path(metadata_db))
        
        self.downloader = BinanceDataDownloader(output_dir=self.raw_dir, max_concurrent=10)
    
    async def download_and_process(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        kline_interval: str = "1h",
        oi_interval: str = "5m",
    ) -> dict[str, int]:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        missing_klines = self.ledger.get_missing_days("klines", symbol, kline_interval, start, end)
        missing_oi = self.ledger.get_missing_days("open_interest", symbol, oi_interval, start, end)
        
        print(f"Missing: {len(missing_klines)} kline days, {len(missing_oi)} OI days")
        
        results = {"klines": 0, "open_interest": 0}
        
        if missing_klines:
            kline_dates = [d.strftime("%Y-%m-%d") for d in missing_klines]
            await self._download_and_convert_klines(symbol, kline_dates, kline_interval)
            results["klines"] = len(missing_klines)
        
        if missing_oi:
            oi_dates = [d.strftime("%Y-%m-%d") for d in missing_oi]
            await self._download_and_convert_oi(symbol, oi_dates, oi_interval)
            results["open_interest"] = len(missing_oi)
        
        return results
    
    async def _download_and_convert_klines(self, symbol: str, dates: list[str], interval: str):
        for date_str in dates:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            key = PartitionKey.from_date(d, "klines", symbol, interval)
            
            files = await self.downloader.download_klines(symbol, date_str, date_str, interval)
            if not files:
                continue
            
            df = self._read_zip_csv(files[0], self.KLINE_COLUMNS)
            if df.is_empty():
                continue
            
            df = df.with_columns([
                (pl.col("open_time").cast(pl.Int64) * 1000).cast(pl.Datetime("us")).alias("timestamp"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.lit(symbol).alias("symbol"),
            ]).select(["timestamp", "symbol", "open", "high", "low", "close", "volume"])
            
            self._write_partition(df, key)
            self.ledger.mark_complete(key, len(df))
            print(f"  {date_str}: {len(df)} klines")
    
    async def _download_and_convert_oi(self, symbol: str, dates: list[str], interval: str):
        for date_str in dates:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            key = PartitionKey.from_date(d, "open_interest", symbol, interval)
            
            files = await self.downloader.download_open_interest(symbol, date_str, date_str, interval)
            if not files:
                continue
            
            df = self._read_zip_csv(files[0], self.OI_COLUMNS)
            if df.is_empty():
                continue
            
            # Handle create_time: can be unix timestamp (int) or datetime string
            create_time_col = df["create_time"]
            if create_time_col.dtype == pl.Utf8:
                # Try parsing as datetime string first
                try:
                    df = df.with_columns([
                        pl.col("create_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("timestamp"),
                        pl.col("sum_open_interest").cast(pl.Float64),
                        pl.col("sum_open_interest_value").cast(pl.Float64),
                    ])
                except Exception:
                    # Fallback: try as unix timestamp string
                    df = df.with_columns([
                        pl.col("create_time").cast(pl.Int64).cast(pl.Datetime("ms")).alias("timestamp"),
                        pl.col("sum_open_interest").cast(pl.Float64),
                        pl.col("sum_open_interest_value").cast(pl.Float64),
                    ])
            else:
                df = df.with_columns([
                    pl.col("create_time").cast(pl.Int64).cast(pl.Datetime("ms")).alias("timestamp"),
                    pl.col("sum_open_interest").cast(pl.Float64),
                    pl.col("sum_open_interest_value").cast(pl.Float64),
                ])
            df = df.select(["timestamp", "symbol", "sum_open_interest", "sum_open_interest_value"])
            
            self._write_partition(df, key)
            self.ledger.mark_complete(key, len(df))
            print(f"  {date_str}: {len(df)} OI rows")
    
    def _read_zip_csv(self, zip_path: Path, columns: list[str]) -> pl.DataFrame:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                with zf.open(csv_name) as f:
                    content = f.read()
                    first_line = content.decode("utf-8").split("\n")[0]
                    has_header = not first_line[0].isdigit()
                    
                    if has_header:
                        return pl.read_csv(io.BytesIO(content))
                    else:
                        return pl.read_csv(io.BytesIO(content), has_header=False, new_columns=columns)
        except Exception as e:
            print(f"Error reading {zip_path}: {e}")
            return pl.DataFrame()
    
    def _write_partition(self, df: pl.DataFrame, key: PartitionKey):
        path = self.silver_dir / key.path / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd")
    
    def read_silver(
        self,
        dataset: str,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        dfs = []
        current = start
        while current <= end:
            key = PartitionKey.from_date(current, dataset, symbol, interval)
            path = self.silver_dir / key.path / "data.parquet"
            if path.exists():
                dfs.append(pl.read_parquet(path))
            current += timedelta(days=1)
        
        if not dfs:
            return pl.DataFrame()
        return pl.concat(dfs).sort("timestamp")
    
    def close(self):
        self.ledger.close()
