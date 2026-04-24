from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable
import logging

from .config import DEFAULT_BARRIER, DEFAULT_POSITION, DEFAULT_TRADING, BarrierConfig, PositionConfig


class PositionSide(Enum):
    NONE = 0
    LONG = 1
    SHORT = -1


@dataclass
class Position:
    side: PositionSide = PositionSide.NONE
    entry_price: float = 0.0
    entry_time: datetime | None = None
    size: float = 0.0
    
    def is_open(self) -> bool:
        return self.side != PositionSide.NONE


@dataclass
class TradeResult:
    entry_time: datetime
    exit_time: datetime
    side: PositionSide
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class RiskState:
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    is_halted: bool = False
    halt_reason: str = ""


class RiskManager:
    
    def __init__(
        self,
        initial_capital: float,
        daily_loss_limit: float = 0.02,
        weekly_loss_limit: float = 0.05,
        max_positions: int = 1,
        barrier_config: BarrierConfig = DEFAULT_BARRIER,
        position_config: PositionConfig = DEFAULT_POSITION,
    ):
        self.initial_capital = initial_capital
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.max_positions = max_positions
        self.barrier = barrier_config
        self.position_cfg = position_config
        
        self.state = RiskState(
            peak_equity=initial_capital,
            current_equity=initial_capital,
        )
        
        self.current_position = Position()
        self.trade_history: list[TradeResult] = []
        self.last_daily_reset: datetime | None = None
        self.last_weekly_reset: datetime | None = None
        
        self.logger = logging.getLogger(__name__)
    
    def can_open_position(self) -> tuple[bool, str]:
        if self.state.is_halted:
            return False, f"Trading halted: {self.state.halt_reason}"
        
        if self.current_position.is_open():
            return False, "Position already open"
        
        return True, ""
    
    def calculate_position_size(self, price: float) -> float:
        equity = self.state.current_equity
        position_value = equity * self.position_cfg.position_size_pct
        return position_value / price
    
    def check_barrier_exit(self, current_price: float, current_time: datetime) -> tuple[bool, str]:
        if not self.current_position.is_open():
            return False, ""
        
        pos = self.current_position
        
        if pos.side == PositionSide.LONG:
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price
        
        if pnl_pct >= self.barrier.profit_take:
            return True, "take_profit"
        
        if pnl_pct <= -self.barrier.stop_loss:
            return True, "stop_loss"
        
        if pos.entry_time:
            hours_held = (current_time - pos.entry_time).total_seconds() / 3600
            if hours_held >= self.barrier.horizon_hours:
                return True, "horizon"
        
        return False, ""
    
    def open_position(self, side: PositionSide, price: float, time: datetime) -> Position:
        size = self.calculate_position_size(price)
        
        fee = size * price * self.position_cfg.taker_fee_pct
        slippage = size * price * (self.position_cfg.slippage_bps / 10000)
        
        adjusted_price = price * (1 + self.position_cfg.slippage_bps / 10000) if side == PositionSide.LONG else price * (1 - self.position_cfg.slippage_bps / 10000)
        
        self.current_position = Position(
            side=side,
            entry_price=adjusted_price,
            entry_time=time,
            size=size,
        )
        
        self.state.current_equity -= fee
        
        self.logger.info(f"Opened {side.name} @ {adjusted_price:.2f}, size={size:.6f}")
        
        return self.current_position
    
    def close_position(self, price: float, time: datetime, reason: str) -> TradeResult:
        pos = self.current_position
        
        if pos.side == PositionSide.LONG:
            adjusted_price = price * (1 - self.position_cfg.slippage_bps / 10000)
            pnl = (adjusted_price - pos.entry_price) * pos.size
        else:
            adjusted_price = price * (1 + self.position_cfg.slippage_bps / 10000)
            pnl = (pos.entry_price - adjusted_price) * pos.size
        
        fee = pos.size * price * self.position_cfg.taker_fee_pct
        pnl -= fee
        
        pnl_pct = pnl / (pos.entry_price * pos.size)
        
        result = TradeResult(
            entry_time=pos.entry_time or time,
            exit_time=time,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=adjusted_price,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        
        self.trade_history.append(result)
        self.state.current_equity += pnl
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity
        
        drawdown = (self.state.peak_equity - self.state.current_equity) / self.state.peak_equity
        if drawdown > self.state.max_drawdown:
            self.state.max_drawdown = drawdown
        
        self._check_limits()
        
        self.current_position = Position()
        
        self.logger.info(f"Closed {pos.side.name} @ {adjusted_price:.2f}, PnL={pnl:.2f} ({pnl_pct:.2%}), reason={reason}")
        
        return result
    
    def _check_limits(self) -> None:
        daily_loss_pct = -self.state.daily_pnl / self.initial_capital
        if daily_loss_pct >= self.daily_loss_limit:
            self.state.is_halted = True
            self.state.halt_reason = f"Daily loss limit hit: {daily_loss_pct:.2%}"
            self.logger.warning(self.state.halt_reason)
        
        weekly_loss_pct = -self.state.weekly_pnl / self.initial_capital
        if weekly_loss_pct >= self.weekly_loss_limit:
            self.state.is_halted = True
            self.state.halt_reason = f"Weekly loss limit hit: {weekly_loss_pct:.2%}"
            self.logger.warning(self.state.halt_reason)
    
    def reset_daily(self, time: datetime) -> None:
        if self.last_daily_reset is None or (time - self.last_daily_reset) >= timedelta(days=1):
            self.state.daily_pnl = 0.0
            self.last_daily_reset = time
            if self.state.is_halted and "Daily" in self.state.halt_reason:
                self.state.is_halted = False
                self.state.halt_reason = ""
    
    def reset_weekly(self, time: datetime) -> None:
        if self.last_weekly_reset is None or (time - self.last_weekly_reset) >= timedelta(weeks=1):
            self.state.weekly_pnl = 0.0
            self.last_weekly_reset = time
            if self.state.is_halted and "Weekly" in self.state.halt_reason:
                self.state.is_halted = False
                self.state.halt_reason = ""
    
    def get_metrics(self) -> dict:
        if not self.trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": self.state.max_drawdown,
            }
        
        wins = sum(1 for t in self.trade_history if t.pnl > 0)
        total_pnl = sum(t.pnl for t in self.trade_history)
        pnl_pcts = [t.pnl_pct for t in self.trade_history]
        
        import numpy as np
        pnl_arr = np.array(pnl_pcts)
        sharpe = (pnl_arr.mean() / pnl_arr.std() * np.sqrt(365 * 24)) if pnl_arr.std() > 0 else 0.0
        
        return {
            "total_trades": len(self.trade_history),
            "win_rate": wins / len(self.trade_history),
            "total_pnl": total_pnl,
            "total_return": (self.state.current_equity - self.initial_capital) / self.initial_capital,
            "sharpe_ratio": float(sharpe),
            "max_drawdown": self.state.max_drawdown,
        }
