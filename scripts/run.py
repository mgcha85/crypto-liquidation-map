#!/usr/bin/env python3
"""
CLI for Crypto Liquidation Map

Usage:
    python -m scripts.run download BTCUSDT 2024-01-01 2024-01-31
    python -m scripts.run process
    python -m scripts.run analyze BTCUSDT --output liquidation_map.html
    python -m scripts.run pipeline BTCUSDT 2024-01-01 2024-01-31
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from liquidation_map.data.downloader import BinanceDataDownloader
from liquidation_map.data.processor import DataProcessor
from liquidation_map.analysis.liquidation_map import LiquidationMapCalculator
from liquidation_map.visualization.heatmap import LiquidationHeatmap


def cmd_download(args):
    async def _download():
        downloader = BinanceDataDownloader(
            output_dir=args.output_dir,
            max_concurrent=args.concurrent,
        )
        
        result = await downloader.download_all(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            kline_interval=args.kline_interval,
            oi_interval=args.oi_interval,
        )
        
        print(f"\nDownload complete:")
        print(f"  Klines: {len(result['klines'])} files")
        print(f"  Metrics (OI): {len(result['open_interest'])} files")
        print(f"  Liquidation: {len(result['liquidation'])} files")
    
    asyncio.run(_download())


def cmd_process(args):
    with DataProcessor(db_path=args.db_path) as processor:
        result = processor.process_all(args.raw_dir, symbol=args.symbol)
        
        print(f"\nProcessing complete:")
        print(f"  Klines: {result['klines']} rows")
        print(f"  Metrics (OI): {result['metrics']} rows")
        print(f"  Liquidation: {result['liquidation']} rows")


def cmd_analyze(args):
    with DataProcessor(db_path=args.db_path) as processor:
        df_oi = processor.query(f"""
            SELECT * FROM open_interest 
            WHERE symbol = '{args.symbol}'
            ORDER BY timestamp
        """)
        
        df_klines = processor.query(f"""
            SELECT * FROM klines 
            WHERE symbol = '{args.symbol}'
            ORDER BY timestamp
        """)
        
        if df_oi.is_empty() or df_klines.is_empty():
            print(f"No data found for {args.symbol}")
            return
        
        print(f"Loaded {len(df_oi)} OI rows, {len(df_klines)} kline rows")
        
        calc = LiquidationMapCalculator(price_bucket_size=args.bucket_size)
        current_price = df_klines.select("close").to_series()[-1]
        
        df_map = calc.calculate(df_oi, df_klines, current_price)
        print(f"Generated {len(df_map)} price buckets")
        
        viz = LiquidationHeatmap()
        fig = viz.create_bar_chart(df_map, current_price, symbol=args.symbol)
        
        output = Path(args.output)
        viz.save(fig, output)
        print(f"Saved to {output}")


def cmd_pipeline(args):
    print(f"=== Full Pipeline for {args.symbol} ===\n")
    
    print("Step 1: Downloading data...")
    cmd_download(args)
    
    print("\nStep 2: Processing data...")
    cmd_process(args)
    
    print("\nStep 3: Analyzing and visualizing...")
    cmd_analyze(args)
    
    print("\n=== Pipeline complete ===")


def main():
    parser = argparse.ArgumentParser(description="Crypto Liquidation Map CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    dl = subparsers.add_parser("download", help="Download data from Binance Vision")
    dl.add_argument("symbol", help="Trading pair (e.g., BTCUSDT)")
    dl.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    dl.add_argument("end_date", help="End date (YYYY-MM-DD)")
    dl.add_argument("--output-dir", default="data/raw", help="Output directory")
    dl.add_argument("--concurrent", type=int, default=5, help="Max concurrent downloads")
    dl.add_argument("--kline-interval", default="1h", help="Kline interval")
    dl.add_argument("--oi-interval", default="5m", help="Open Interest interval")
    dl.set_defaults(func=cmd_download)
    
    pr = subparsers.add_parser("process", help="Process raw ZIP files into DuckDB")
    pr.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    pr.add_argument("--db-path", default="data/liquidation.duckdb", help="DuckDB path")
    pr.add_argument("--symbol", default=None, help="Filter by symbol")
    pr.set_defaults(func=cmd_process)
    
    an = subparsers.add_parser("analyze", help="Generate liquidation map")
    an.add_argument("symbol", help="Trading pair (e.g., BTCUSDT)")
    an.add_argument("--db-path", default="data/liquidation.duckdb", help="DuckDB path")
    an.add_argument("--output", default="output/liquidation_map.html", help="Output file")
    an.add_argument("--bucket-size", type=float, default=100, help="Price bucket size")
    an.set_defaults(func=cmd_analyze)
    
    pl = subparsers.add_parser("pipeline", help="Full pipeline: download → process → analyze")
    pl.add_argument("symbol", help="Trading pair (e.g., BTCUSDT)")
    pl.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    pl.add_argument("end_date", help="End date (YYYY-MM-DD)")
    pl.add_argument("--output-dir", default="data/raw")
    pl.add_argument("--raw-dir", default="data/raw")
    pl.add_argument("--db-path", default="data/liquidation.duckdb")
    pl.add_argument("--output", default="output/liquidation_map.html")
    pl.add_argument("--concurrent", type=int, default=5)
    pl.add_argument("--kline-interval", default="1h")
    pl.add_argument("--oi-interval", default="5m")
    pl.add_argument("--bucket-size", type=float, default=100)
    pl.set_defaults(func=cmd_pipeline)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
