import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime, timedelta

from live_trading.src.risk_manager import RiskManager, PositionSide
from live_trading.src.config import DEFAULT_BARRIER, DEFAULT_POSITION


def test_daily_loss_limit():
    rm = RiskManager(
        initial_capital=10000.0,
        daily_loss_limit=0.02,
    )
    
    rm.open_position(PositionSide.LONG, 100.0, datetime(2026, 1, 1, 10, 0))
    
    rm.close_position(98.0, datetime(2026, 1, 1, 11, 0), "stop_loss")
    
    rm.current_position = rm.current_position.__class__()
    
    rm.state.daily_pnl = -250.0
    rm._check_limits()
    
    assert rm.state.is_halted, "Should halt after daily loss limit"
    assert "Daily" in rm.state.halt_reason
    print("PASS: Daily loss limit test")


def test_max_position():
    rm = RiskManager(initial_capital=10000.0, max_positions=1)
    
    rm.open_position(PositionSide.LONG, 100.0, datetime(2026, 1, 1, 10, 0))
    
    can_open, reason = rm.can_open_position()
    assert not can_open, "Should not allow second position"
    assert "already open" in reason.lower()
    print("PASS: Max position test")


def test_barrier_take_profit():
    rm = RiskManager(initial_capital=10000.0)
    
    rm.open_position(PositionSide.LONG, 100.0, datetime(2026, 1, 1, 10, 0))
    
    should_exit, reason = rm.check_barrier_exit(102.5, datetime(2026, 1, 1, 11, 0))
    assert should_exit, "Should trigger take profit at +2.5%"
    assert reason == "take_profit"
    print("PASS: Take profit barrier test")


def test_barrier_stop_loss():
    rm = RiskManager(initial_capital=10000.0)
    
    rm.open_position(PositionSide.LONG, 100.0, datetime(2026, 1, 1, 10, 0))
    
    should_exit, reason = rm.check_barrier_exit(98.5, datetime(2026, 1, 1, 11, 0))
    assert should_exit, "Should trigger stop loss at -1.5%"
    assert reason == "stop_loss"
    print("PASS: Stop loss barrier test")


def test_barrier_horizon():
    rm = RiskManager(initial_capital=10000.0)
    
    entry_time = datetime(2026, 1, 1, 10, 0)
    rm.open_position(PositionSide.LONG, 100.0, entry_time)
    
    should_exit, reason = rm.check_barrier_exit(100.5, entry_time + timedelta(hours=24))
    assert not should_exit, "Should not exit at 24h"
    
    should_exit, reason = rm.check_barrier_exit(100.5, entry_time + timedelta(hours=49))
    assert should_exit, "Should exit at 49h (past 48h horizon)"
    assert reason == "horizon"
    print("PASS: Horizon barrier test")


def test_weekly_loss_limit():
    rm = RiskManager(
        initial_capital=10000.0,
        weekly_loss_limit=0.05,
    )
    
    rm.state.weekly_pnl = -600.0
    rm._check_limits()
    
    assert rm.state.is_halted, "Should halt after weekly loss limit"
    assert "Weekly" in rm.state.halt_reason
    print("PASS: Weekly loss limit test")


if __name__ == "__main__":
    print("=" * 60)
    print("CP-004: Risk Limits Verification")
    print("=" * 60)
    
    test_daily_loss_limit()
    test_max_position()
    test_barrier_take_profit()
    test_barrier_stop_loss()
    test_barrier_horizon()
    test_weekly_loss_limit()
    
    print()
    print("ALL RISK TESTS PASSED")
