from dataclasses import dataclass
from datetime import datetime
from typing import Callable
import asyncio
import logging

import polars as pl

from .config import DEFAULT_TRADING, TradingConfig
from .features import LiveFeatureExtractor, FeatureWindow
from .model import TradingModel, Signal
from .risk_manager import RiskManager, PositionSide, TradeResult


@dataclass
class EngineState:
    is_running: bool = False
    last_update: datetime | None = None
    last_signal: Signal = 0
    error_count: int = 0


class TradingEngine:
    
    def __init__(
        self,
        config: TradingConfig,
        model: TradingModel,
        risk_manager: RiskManager,
        data_fetcher: Callable[[str, int], tuple[pl.DataFrame, pl.DataFrame]],
    ):
        self.config = config
        self.model = model
        self.risk_manager = risk_manager
        self.fetch_data = data_fetcher
        
        self.feature_extractor = LiveFeatureExtractor()
        self.state = EngineState()
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    
    async def run(self) -> None:
        self.state.is_running = True
        self.logger.info(f"Starting trading engine in {self.config.mode} mode for {self.config.symbol}")
        
        while self.state.is_running:
            try:
                await self._tick()
                await asyncio.sleep(self.config.update_interval_sec)
            except Exception as e:
                self.state.error_count += 1
                self.logger.error(f"Engine error: {e}")
                
                if self.state.error_count >= 3:
                    self.logger.warning("3 consecutive errors - pausing for 5 minutes")
                    await asyncio.sleep(300)
                    self.state.error_count = 0
    
    async def _tick(self) -> None:
        now = datetime.utcnow()
        
        self.risk_manager.reset_daily(now)
        if now.weekday() == 0:
            self.risk_manager.reset_weekly(now)
        
        df_oi, df_klines = await asyncio.to_thread(
            self.fetch_data,
            self.config.symbol,
            self.config.lookback_hours,
        )
        
        if df_klines.is_empty():
            self.logger.warning("No kline data received")
            return
        
        current_price = float(df_klines.sort("open_time")["close"][-1])
        
        should_exit, exit_reason = self.risk_manager.check_barrier_exit(current_price, now)
        if should_exit:
            result = self.risk_manager.close_position(current_price, now, exit_reason)
            self._log_trade(result)
        
        features = self.feature_extractor.extract(df_oi, df_klines, current_price, now)
        
        if self.config.log_features:
            self._log_features(features)
        
        df_features = features.to_dataframe()
        signal = self.model.predict(df_features)
        self.state.last_signal = signal
        
        await self._process_signal(signal, current_price, now)
        
        self.state.last_update = now
        self.state.error_count = 0
    
    async def _process_signal(self, signal: Signal, price: float, time: datetime) -> None:
        current_side = self.risk_manager.current_position.side
        
        if signal == 1 and current_side != PositionSide.LONG:
            if current_side == PositionSide.SHORT:
                result = self.risk_manager.close_position(price, time, "signal_flip")
                self._log_trade(result)
            
            can_open, reason = self.risk_manager.can_open_position()
            if can_open:
                self.risk_manager.open_position(PositionSide.LONG, price, time)
            else:
                self.logger.debug(f"Cannot open LONG: {reason}")
        
        elif signal == -1 and current_side != PositionSide.SHORT:
            if current_side == PositionSide.LONG:
                result = self.risk_manager.close_position(price, time, "signal_flip")
                self._log_trade(result)
            
            can_open, reason = self.risk_manager.can_open_position()
            if can_open:
                self.risk_manager.open_position(PositionSide.SHORT, price, time)
            else:
                self.logger.debug(f"Cannot open SHORT: {reason}")
        
        elif signal == 0 and current_side != PositionSide.NONE:
            result = self.risk_manager.close_position(price, time, "signal_neutral")
            self._log_trade(result)
    
    def _log_trade(self, result: TradeResult) -> None:
        if not self.config.log_trades:
            return
        
        self.logger.info(
            f"TRADE: {result.side.name} | "
            f"Entry: {result.entry_price:.2f} | Exit: {result.exit_price:.2f} | "
            f"PnL: {result.pnl:.2f} ({result.pnl_pct:.2%}) | "
            f"Reason: {result.exit_reason}"
        )
    
    def _log_features(self, features: FeatureWindow) -> None:
        self.logger.debug(f"Features @ {features.timestamp}: {features.features}")
    
    def stop(self) -> None:
        self.state.is_running = False
        self.logger.info("Stopping trading engine")
    
    def get_status(self) -> dict:
        metrics = self.risk_manager.get_metrics()
        return {
            "is_running": self.state.is_running,
            "mode": self.config.mode,
            "symbol": self.config.symbol,
            "last_update": self.state.last_update.isoformat() if self.state.last_update else None,
            "last_signal": int(self.state.last_signal),
            "position": self.risk_manager.current_position.side.name,
            "is_halted": self.risk_manager.state.is_halted,
            "halt_reason": self.risk_manager.state.halt_reason,
            **metrics,
        }
