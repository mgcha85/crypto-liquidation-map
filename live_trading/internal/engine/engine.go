package engine

import (
	"context"
	"math"
	"sync"
	"time"

	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/binance"
	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/config"
	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/model"
	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/risk"
	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/telegram"
	"github.com/rs/zerolog/log"
)

type Signal int

const (
	SignalSell Signal = -1
	SignalHold Signal = 0
	SignalBuy  Signal = 1
)

type State struct {
	IsRunning   bool
	LastUpdate  time.Time
	LastSignal  Signal
	ErrorCount  int
}

type Engine struct {
	mu sync.RWMutex

	cfg       *config.Config
	client    *binance.Client
	predictor *model.Predictor
	riskMgr   *risk.Manager
	telegram  *telegram.Client

	state  State
	cancel context.CancelFunc
}

func New(cfg *config.Config, initialCapital float64) (*Engine, error) {
	predictor, err := model.NewPredictor(cfg.ModelPath)
	if err != nil {
		return nil, err
	}

	tg := telegram.New(cfg.TelegramBotToken, cfg.TelegramChatID, cfg.TelegramEnabled)

	return &Engine{
		cfg:       cfg,
		client:    binance.NewClient(&cfg.API),
		predictor: predictor,
		riskMgr:   risk.NewManager(initialCapital, cfg),
		telegram:  tg,
	}, nil
}

func (e *Engine) Start(ctx context.Context) {
	ctx, e.cancel = context.WithCancel(ctx)

	e.mu.Lock()
	e.state.IsRunning = true
	e.mu.Unlock()

	log.Info().
		Str("mode", e.cfg.Mode).
		Str("symbol", e.cfg.Symbol).
		Msg("Starting trading engine")

	ticker := time.NewTicker(time.Duration(e.cfg.UpdateIntervalSec) * time.Second)
	defer ticker.Stop()

	e.tick()

	for {
		select {
		case <-ctx.Done():
			log.Info().Msg("Engine stopped")
			return
		case <-ticker.C:
			e.tick()
		}
	}
}

func (e *Engine) Stop() {
	if e.cancel != nil {
		e.cancel()
	}
	e.mu.Lock()
	e.state.IsRunning = false
	e.mu.Unlock()
}

func (e *Engine) tick() {
	now := time.Now().UTC()

	e.riskMgr.ResetDaily(now)
	if now.Weekday() == time.Monday {
		e.riskMgr.ResetWeekly(now)
	}

	klines, err := e.client.GetKlines(e.cfg.Symbol, "1h", e.cfg.LookbackHours)
	if err != nil {
		log.Error().Err(err).Msg("Failed to fetch klines")
		e.incrementError()
		return
	}

	if len(klines) == 0 {
		log.Warn().Msg("No klines received")
		return
	}

	currentPrice := klines[len(klines)-1].Close

	shouldExit, reason := e.riskMgr.CheckBarrierExit(currentPrice, now)
	if shouldExit {
		trade := e.riskMgr.ClosePosition(currentPrice, now, reason)
		e.executeClose(trade)
	}

	ois, err := e.client.GetOpenInterestHist(e.cfg.Symbol, "1h", e.cfg.LookbackHours)
	if err != nil {
		log.Warn().Err(err).Msg("Failed to fetch OI data")
	}

	features := e.extractFeatures(klines, ois, currentPrice)

	signal, confidence, err := e.predictor.Predict(features)
	if err != nil {
		log.Error().Err(err).Msg("Prediction failed")
		e.incrementError()
		return
	}

	e.mu.Lock()
	e.state.LastSignal = Signal(signal)
	e.state.LastUpdate = now
	e.state.ErrorCount = 0
	e.mu.Unlock()

	log.Info().
		Float64("price", currentPrice).
		Int("signal", signal).
		Float64("confidence", confidence).
		Msg("Tick completed")

	e.processSignal(Signal(signal), currentPrice, now)
}

func (e *Engine) processSignal(signal Signal, price float64, now time.Time) {
	pos := e.riskMgr.GetPosition()

	if signal == SignalBuy && pos.Side != risk.SideLong {
		if pos.Side == risk.SideShort {
			trade := e.riskMgr.ClosePosition(price, now, "signal_flip")
			e.executeClose(trade)
		}

		if ok, reason := e.riskMgr.CanOpenPosition(); ok {
			e.riskMgr.OpenPosition(risk.SideLong, price, now)
			e.executeOpen("BUY", price)
		} else {
			log.Debug().Str("reason", reason).Msg("Cannot open LONG")
		}
	} else if signal == SignalSell && pos.Side != risk.SideShort {
		if pos.Side == risk.SideLong {
			trade := e.riskMgr.ClosePosition(price, now, "signal_flip")
			e.executeClose(trade)
		}

		if ok, reason := e.riskMgr.CanOpenPosition(); ok {
			e.riskMgr.OpenPosition(risk.SideShort, price, now)
			e.executeOpen("SELL", price)
		} else {
			log.Debug().Str("reason", reason).Msg("Cannot open SHORT")
		}
	} else if signal == SignalHold && pos.Side != risk.SideNone {
		trade := e.riskMgr.ClosePosition(price, now, "signal_neutral")
		e.executeClose(trade)
	}
}

func (e *Engine) executeOpen(side string, price float64) {
	if e.cfg.Mode == "paper" {
		log.Info().Str("side", side).Float64("price", price).Msg("[PAPER] Open position")
		return
	}

	size := e.riskMgr.CalcPositionSize(price)
	result, err := e.client.PlaceOrder(e.cfg.Symbol, side, size, false)
	if err != nil {
		log.Error().Err(err).Msg("Failed to place order")
		return
	}

	log.Info().
		Int64("orderId", result.OrderID).
		Str("side", result.Side).
		Float64("qty", result.Qty).
		Msg("Order placed")
}

func (e *Engine) executeClose(trade risk.Trade) {
	if e.cfg.Mode == "paper" {
		log.Info().
			Str("side", trade.Side.String()).
			Float64("pnl", trade.PnL).
			Str("reason", trade.ExitReason).
			Msg("[PAPER] Close position")

		e.telegram.SendTradeNotification(
			trade.Side.String(),
			trade.EntryPrice,
			trade.ExitPrice,
			trade.PnL,
			trade.PnLPct,
		)
		return
	}

	result, err := e.client.ClosePosition(e.cfg.Symbol)
	if err != nil {
		log.Error().Err(err).Msg("Failed to close position")
		return
	}

	if result != nil {
		log.Info().
			Int64("orderId", result.OrderID).
			Float64("qty", result.Qty).
			Msg("Position closed")

		e.telegram.SendTradeNotification(
			trade.Side.String(),
			trade.EntryPrice,
			trade.ExitPrice,
			trade.PnL,
			trade.PnLPct,
		)
	}
}

func (e *Engine) incrementError() {
	e.mu.Lock()
	e.state.ErrorCount++
	e.mu.Unlock()
}

func (e *Engine) extractFeatures(klines []binance.Kline, ois []binance.OpenInterest, currentPrice float64) []float64 {
	features := make([]float64, 31)

	if len(ois) > 0 {
		var totalOI, longOI, shortOI float64
		for _, o := range ois {
			totalOI += o.OIValue
		}
		longOI = totalOI * 0.5
		shortOI = totalOI * 0.5

		features[0] = totalOI
		features[1] = longOI
		features[2] = shortOI
		features[3] = safeDiv(longOI, shortOI)
		features[4] = 1.0
		features[5] = 0.1
		features[6] = 0.2
		features[7] = 0.5
		features[8] = 0.02
		features[9] = 0.02
		features[10] = totalOI * 0.1
		features[11] = totalOI * 0.1
		features[12] = 0.01
		features[13] = 0.02
		features[14] = 0.03
		features[15] = 0.01
		features[16] = 0.02
		features[17] = 0.03
		features[18] = 2.0
		features[19] = 0.0
	}

	n := len(klines)
	if n >= 2 {
		features[20] = (klines[n-1].Close / klines[n-2].Close) - 1
	}
	if n >= 7 {
		features[21] = (klines[n-1].Close / klines[n-7].Close) - 1
	}
	if n >= 13 {
		features[22] = (klines[n-1].Close / klines[n-13].Close) - 1
	}
	if n >= 25 {
		features[23] = (klines[n-1].Close / klines[n-25].Close) - 1
	}

	if n >= 7 {
		features[24] = calcVolatility(klines[n-7:])
	}
	if n >= 25 {
		features[25] = calcVolatility(klines[n-25:])
	}

	if n >= 24 {
		features[26] = calcATR(klines[n-24:])
	}

	if n >= 24 {
		var volSum float64
		for i := n - 24; i < n; i++ {
			volSum += klines[i].Volume
		}
		volMA := volSum / 24
		features[27] = safeDiv(klines[n-1].Volume, volMA)
	}

	last := klines[n-1]
	totalRange := last.High - last.Low
	if totalRange > 0 {
		features[28] = (last.High - math.Max(last.Close, last.Open)) / totalRange
		features[29] = (math.Min(last.Close, last.Open) - last.Low) / totalRange
	}

	if n >= 24 {
		high24 := klines[n-24].High
		low24 := klines[n-24].Low
		for i := n - 24; i < n; i++ {
			if klines[i].High > high24 {
				high24 = klines[i].High
			}
			if klines[i].Low < low24 {
				low24 = klines[i].Low
			}
		}
		features[30] = safeDiv(last.Close-low24, high24-low24)
	}

	return features
}

func calcVolatility(klines []binance.Kline) float64 {
	if len(klines) < 2 {
		return 0
	}

	returns := make([]float64, len(klines)-1)
	for i := 1; i < len(klines); i++ {
		returns[i-1] = (klines[i].Close / klines[i-1].Close) - 1
	}

	var sum, sumSq float64
	for _, r := range returns {
		sum += r
		sumSq += r * r
	}
	mean := sum / float64(len(returns))
	variance := (sumSq / float64(len(returns))) - (mean * mean)
	return math.Sqrt(variance)
}

func calcATR(klines []binance.Kline) float64 {
	if len(klines) < 2 {
		return 0
	}

	var sum float64
	for i := 1; i < len(klines); i++ {
		tr := math.Max(
			klines[i].High-klines[i].Low,
			math.Max(
				math.Abs(klines[i].High-klines[i-1].Close),
				math.Abs(klines[i].Low-klines[i-1].Close),
			),
		)
		sum += tr
	}
	return sum / float64(len(klines)-1)
}

func safeDiv(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func (e *Engine) GetState() State {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.state
}

func (e *Engine) GetRiskManager() *risk.Manager {
	return e.riskMgr
}

func (e *Engine) Close() error {
	return e.predictor.Close()
}

func (e *Engine) SetTelegramEnabled(enabled bool) {
	e.telegram.SetEnabled(enabled)
}

func (e *Engine) IsTelegramEnabled() bool {
	return e.telegram.IsEnabled()
}
