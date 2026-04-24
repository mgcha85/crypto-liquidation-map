package risk

import (
	"sync"
	"time"

	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/config"
)

type PositionSide int

const (
	SideNone PositionSide = iota
	SideLong
	SideShort
)

func (s PositionSide) String() string {
	switch s {
	case SideLong:
		return "LONG"
	case SideShort:
		return "SHORT"
	default:
		return "NONE"
	}
}

type Position struct {
	Side       PositionSide
	EntryPrice float64
	EntryTime  time.Time
	Size       float64
}

func (p *Position) IsOpen() bool {
	return p.Side != SideNone
}

type Trade struct {
	ID         string
	EntryTime  time.Time
	ExitTime   time.Time
	Side       PositionSide
	EntryPrice float64
	ExitPrice  float64
	Size       float64
	PnL        float64
	PnLPct     float64
	ExitReason string
}

type State struct {
	DailyPnL      float64
	WeeklyPnL     float64
	MaxDrawdown   float64
	PeakEquity    float64
	CurrentEquity float64
	IsHalted      bool
	HaltReason    string
}

type Manager struct {
	mu sync.RWMutex

	initialCapital float64
	cfg            *config.Config

	state    State
	position Position
	trades   []Trade

	lastDailyReset  time.Time
	lastWeeklyReset time.Time
}

func NewManager(initialCapital float64, cfg *config.Config) *Manager {
	return &Manager{
		initialCapital: initialCapital,
		cfg:            cfg,
		state: State{
			PeakEquity:    initialCapital,
			CurrentEquity: initialCapital,
		},
	}
}

func (m *Manager) CanOpenPosition() (bool, string) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.state.IsHalted {
		return false, "trading halted: " + m.state.HaltReason
	}
	if m.position.IsOpen() {
		return false, "position already open"
	}
	return true, ""
}

func (m *Manager) CalcPositionSize(price float64) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	positionValue := m.state.CurrentEquity * m.cfg.Position.SizePct
	return positionValue / price
}

func (m *Manager) CheckBarrierExit(currentPrice float64, now time.Time) (bool, string) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.position.IsOpen() {
		return false, ""
	}

	var pnlPct float64
	if m.position.Side == SideLong {
		pnlPct = (currentPrice - m.position.EntryPrice) / m.position.EntryPrice
	} else {
		pnlPct = (m.position.EntryPrice - currentPrice) / m.position.EntryPrice
	}

	if pnlPct >= m.cfg.Barrier.ProfitTake {
		return true, "take_profit"
	}
	if pnlPct <= -m.cfg.Barrier.StopLoss {
		return true, "stop_loss"
	}

	hoursHeld := now.Sub(m.position.EntryTime).Hours()
	if hoursHeld >= float64(m.cfg.Barrier.HorizonHours) {
		return true, "horizon"
	}

	return false, ""
}

func (m *Manager) OpenPosition(side PositionSide, price float64, now time.Time) Position {
	m.mu.Lock()
	defer m.mu.Unlock()

	size := m.state.CurrentEquity * m.cfg.Position.SizePct / price
	slippage := m.cfg.Position.SlippageBps / 10000.0

	var adjustedPrice float64
	if side == SideLong {
		adjustedPrice = price * (1 + slippage)
	} else {
		adjustedPrice = price * (1 - slippage)
	}

	fee := size * price * m.cfg.Position.TakerFeePct
	m.state.CurrentEquity -= fee

	m.position = Position{
		Side:       side,
		EntryPrice: adjustedPrice,
		EntryTime:  now,
		Size:       size,
	}

	return m.position
}

func (m *Manager) ClosePosition(price float64, now time.Time, reason string) Trade {
	m.mu.Lock()
	defer m.mu.Unlock()

	pos := m.position
	slippage := m.cfg.Position.SlippageBps / 10000.0

	var adjustedPrice, pnl float64
	if pos.Side == SideLong {
		adjustedPrice = price * (1 - slippage)
		pnl = (adjustedPrice - pos.EntryPrice) * pos.Size
	} else {
		adjustedPrice = price * (1 + slippage)
		pnl = (pos.EntryPrice - adjustedPrice) * pos.Size
	}

	fee := pos.Size * price * m.cfg.Position.TakerFeePct
	pnl -= fee

	pnlPct := pnl / (pos.EntryPrice * pos.Size)

	trade := Trade{
		ID:         now.Format("20060102150405"),
		EntryTime:  pos.EntryTime,
		ExitTime:   now,
		Side:       pos.Side,
		EntryPrice: pos.EntryPrice,
		ExitPrice:  adjustedPrice,
		Size:       pos.Size,
		PnL:        pnl,
		PnLPct:     pnlPct,
		ExitReason: reason,
	}

	m.trades = append(m.trades, trade)
	m.state.CurrentEquity += pnl
	m.state.DailyPnL += pnl
	m.state.WeeklyPnL += pnl

	if m.state.CurrentEquity > m.state.PeakEquity {
		m.state.PeakEquity = m.state.CurrentEquity
	}

	drawdown := (m.state.PeakEquity - m.state.CurrentEquity) / m.state.PeakEquity
	if drawdown > m.state.MaxDrawdown {
		m.state.MaxDrawdown = drawdown
	}

	m.checkLimits()
	m.position = Position{}

	return trade
}

func (m *Manager) checkLimits() {
	dailyLossPct := -m.state.DailyPnL / m.initialCapital
	if dailyLossPct >= m.cfg.Risk.DailyLossLimitPct {
		m.state.IsHalted = true
		m.state.HaltReason = "daily loss limit hit"
	}

	weeklyLossPct := -m.state.WeeklyPnL / m.initialCapital
	if weeklyLossPct >= m.cfg.Risk.WeeklyLossLimitPct {
		m.state.IsHalted = true
		m.state.HaltReason = "weekly loss limit hit"
	}
}

func (m *Manager) ResetDaily(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.lastDailyReset.IsZero() || now.Sub(m.lastDailyReset) >= 24*time.Hour {
		m.state.DailyPnL = 0
		m.lastDailyReset = now
		if m.state.IsHalted && m.state.HaltReason == "daily loss limit hit" {
			m.state.IsHalted = false
			m.state.HaltReason = ""
		}
	}
}

func (m *Manager) ResetWeekly(now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.lastWeeklyReset.IsZero() || now.Sub(m.lastWeeklyReset) >= 7*24*time.Hour {
		m.state.WeeklyPnL = 0
		m.lastWeeklyReset = now
		if m.state.IsHalted && m.state.HaltReason == "weekly loss limit hit" {
			m.state.IsHalted = false
			m.state.HaltReason = ""
		}
	}
}

func (m *Manager) GetState() State {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state
}

func (m *Manager) GetPosition() Position {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.position
}

func (m *Manager) GetTrades() []Trade {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return append([]Trade{}, m.trades...)
}

func (m *Manager) GetMetrics() map[string]float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	totalTrades := len(m.trades)
	if totalTrades == 0 {
		return map[string]float64{
			"total_trades":  0,
			"win_rate":      0,
			"total_pnl":     0,
			"total_return":  0,
			"max_drawdown":  m.state.MaxDrawdown,
		}
	}

	wins := 0
	totalPnL := 0.0
	for _, t := range m.trades {
		if t.PnL > 0 {
			wins++
		}
		totalPnL += t.PnL
	}

	return map[string]float64{
		"total_trades":  float64(totalTrades),
		"win_rate":      float64(wins) / float64(totalTrades),
		"total_pnl":     totalPnL,
		"total_return":  (m.state.CurrentEquity - m.initialCapital) / m.initialCapital,
		"max_drawdown":  m.state.MaxDrawdown,
	}
}

func (m *Manager) SetPositionSizePct(pct float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cfg.Position.SizePct = pct / 100.0
}

func (m *Manager) GetPositionSizePct() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.cfg.Position.SizePct * 100.0
}
