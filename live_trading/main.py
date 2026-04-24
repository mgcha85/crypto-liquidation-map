#!/usr/bin/env python3
import argparse
import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live_trading.src.config import TradingConfig, DEFAULT_TRADING
from live_trading.src.model import TradingModel
from live_trading.src.risk_manager import RiskManager
from live_trading.src.engine import TradingEngine
from live_trading.src.executor import BinanceExecutor, create_data_fetcher


def load_config(config_path: str) -> TradingConfig:
    import yaml
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    return TradingConfig(
        mode=cfg.get("mode", "paper"),
        symbol=cfg.get("symbol", "BTCUSDT"),
        lookback_hours=cfg.get("data", {}).get("lookback_hours", 50),
        update_interval_sec=cfg.get("data", {}).get("update_interval_sec", 3600),
        daily_loss_limit_pct=cfg.get("risk", {}).get("daily_loss_limit_pct", 0.02),
        weekly_loss_limit_pct=cfg.get("risk", {}).get("weekly_loss_limit_pct", 0.05),
        max_positions=cfg.get("risk", {}).get("max_positions", 1),
        log_level=cfg.get("logging", {}).get("level", "INFO"),
        log_trades=cfg.get("logging", {}).get("log_trades", True),
        log_features=cfg.get("logging", {}).get("log_features", False),
        model_path=cfg.get("paths", {}).get("model", "models/xgb_optuna_best.json"),
    )


async def main(config_path: str, initial_capital: float):
    config = load_config(config_path)
    
    model = TradingModel(config.model_path)
    
    risk_manager = RiskManager(
        initial_capital=initial_capital,
        daily_loss_limit=config.daily_loss_limit_pct,
        weekly_loss_limit=config.weekly_loss_limit_pct,
        max_positions=config.max_positions,
    )
    
    executor = BinanceExecutor()
    data_fetcher = create_data_fetcher(executor)
    
    engine = TradingEngine(
        config=config,
        model=model,
        risk_manager=risk_manager,
        data_fetcher=data_fetcher,
    )
    
    def shutdown(sig, frame):
        print("\nShutting down...")
        engine.stop()
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    try:
        await engine.run()
    finally:
        await executor.close()
        
        status = engine.get_status()
        print("\nFinal Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trading Engine")
    parser.add_argument(
        "--config",
        default="configs/paper.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital in USDT",
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(args.config, args.capital))
