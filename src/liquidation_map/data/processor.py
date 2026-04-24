"""
Data Processor - ZIP extraction and DuckDB pipeline using Polars.
"""

import io
import zipfile
from pathlib import Path

import duckdb
import polars as pl

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]

METRICS_COLUMNS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
]

LIQUIDATION_COLUMNS = [
    "symbol", "side", "order_type", "time_in_force", "original_quantity",
    "price", "average_price", "order_status", "last_filled_quantity",
    "filled_accumulated_quantity", "time",
]


class DataProcessor:
    """Process raw Binance ZIP files into DuckDB for analysis."""
    
    def __init__(self, db_path: str | Path = "data/liquidation.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()
    
    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS open_interest (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                sum_open_interest DOUBLE,
                sum_open_interest_value DOUBLE
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS liquidation_events (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                side VARCHAR,
                price DOUBLE,
                quantity DOUBLE,
                quote_quantity DOUBLE
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                quote_volume DOUBLE,
                trades BIGINT
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_oi_symbol_ts ON open_interest(symbol, timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_klines_symbol_ts ON klines(symbol, timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_liq_symbol_ts ON liquidation_events(symbol, timestamp)
        """)
    
    def _extract_symbol_from_path(self, file_path: Path) -> str:
        return file_path.stem.split("-")[0]
    
    def _read_csv_from_zip(
        self, 
        zip_path: Path, 
        columns: list[str] | None = None,
        has_header: bool = False,
    ) -> pl.DataFrame:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                content = f.read()
                if has_header:
                    return pl.read_csv(io.BytesIO(content), infer_schema_length=1000)
                else:
                    return pl.read_csv(
                        io.BytesIO(content),
                        has_header=False,
                        new_columns=columns,
                        infer_schema_length=1000,
                    )
    
    def process_klines(self, raw_dir: str | Path, symbol: str | None = None) -> int:
        raw_path = Path(raw_dir)
        zip_files = list(raw_path.glob("**/*.zip"))
        
        if not zip_files:
            print(f"No ZIP files found in {raw_path}")
            return 0
        
        total_rows = 0
        for zip_path in zip_files:
            file_symbol = self._extract_symbol_from_path(zip_path)
            if symbol and file_symbol != symbol.upper():
                continue
            
            try:
                df = self._read_csv_from_zip(zip_path, has_header=True)
                
                df = df.select([
                    pl.lit(file_symbol).alias("symbol"),
                    pl.col("open_time").cast(pl.Datetime("ms")).alias("timestamp"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                    pl.col("quote_volume").cast(pl.Float64),
                    pl.col("count").cast(pl.Int64).alias("trades"),
                ])
                
                self.conn.execute("INSERT INTO klines SELECT * FROM df")
                total_rows += len(df)
            except Exception as e:
                print(f"Error processing {zip_path}: {e}")
        
        print(f"Inserted {total_rows} kline rows")
        return total_rows
    
    def process_metrics(self, raw_dir: str | Path, symbol: str | None = None) -> int:
        raw_path = Path(raw_dir)
        zip_files = list(raw_path.glob("**/*.zip"))
        
        if not zip_files:
            print(f"No ZIP files found in {raw_path}")
            return 0
        
        total_rows = 0
        for zip_path in zip_files:
            file_symbol = self._extract_symbol_from_path(zip_path)
            if symbol and file_symbol != symbol.upper():
                continue
            
            try:
                df = self._read_csv_from_zip(zip_path, has_header=True)
                
                df = df.select([
                    pl.col("symbol"),
                    pl.col("create_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("timestamp"),
                    pl.col("sum_open_interest").cast(pl.Float64),
                    pl.col("sum_open_interest_value").cast(pl.Float64),
                ])
                
                self.conn.execute("INSERT INTO open_interest SELECT * FROM df")
                total_rows += len(df)
            except Exception as e:
                print(f"Error processing {zip_path}: {e}")
        
        print(f"Inserted {total_rows} metrics rows")
        return total_rows
    
    def process_liquidation_snapshot(self, raw_dir: str | Path, symbol: str | None = None) -> int:
        """Process Liquidation Snapshot ZIP files into DuckDB."""
        raw_path = Path(raw_dir)
        zip_files = list(raw_path.glob("**/*.zip"))
        
        if not zip_files:
            print(f"No ZIP files found in {raw_path}")
            return 0
        
        total_rows = 0
        for zip_path in zip_files:
            file_symbol = self._extract_symbol_from_path(zip_path)
            if symbol and file_symbol != symbol.upper():
                continue
            
            try:
                df = self._read_csv_from_zip(zip_path, LIQUIDATION_COLUMNS)
                
                df = df.select([
                    pl.col("symbol"),
                    pl.col("time").cast(pl.Datetime("ms")).alias("timestamp"),
                    pl.col("side"),
                    pl.col("price").cast(pl.Float64),
                    pl.col("original_quantity").cast(pl.Float64).alias("quantity"),
                    (pl.col("price").cast(pl.Float64) * pl.col("original_quantity").cast(pl.Float64)).alias("quote_quantity"),
                ])
                
                self.conn.execute("INSERT INTO liquidation_events SELECT * FROM df")
                total_rows += len(df)
            except Exception as e:
                print(f"Error processing {zip_path}: {e}")
        
        print(f"Inserted {total_rows} liquidation rows")
        return total_rows
    
    def process_all(self, raw_base_dir: str | Path, symbol: str | None = None) -> dict[str, int]:
        """Process all data types from raw directory structure."""
        base = Path(raw_base_dir)
        
        results = {
            "klines": self.process_klines(base / "klines", symbol),
            "metrics": self.process_metrics(base / "metrics", symbol),
            "liquidation": self.process_liquidation_snapshot(base / "liquidationSnapshot", symbol),
        }
        
        return results
    
    def query(self, sql: str) -> pl.DataFrame:
        """Execute SQL query and return as Polars DataFrame."""
        return self.conn.execute(sql).pl()
    
    def get_symbols(self) -> list[str]:
        """Get list of available symbols in the database."""
        df = self.query("SELECT DISTINCT symbol FROM klines ORDER BY symbol")
        return df["symbol"].to_list()
    
    def get_date_range(self, symbol: str) -> tuple[str, str] | None:
        """Get date range for a symbol."""
        df = self.query(f"""
            SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts 
            FROM klines 
            WHERE symbol = '{symbol}'
        """)
        if df.is_empty():
            return None
        return (str(df["min_ts"][0]), str(df["max_ts"][0]))
    
    def close(self) -> None:
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
