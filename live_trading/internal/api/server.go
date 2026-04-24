package api

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/engine"
	"github.com/rs/zerolog/log"
)

type Server struct {
	engine *engine.Engine
	router *mux.Router
}

func NewServer(eng *engine.Engine) *Server {
	s := &Server{
		engine: eng,
		router: mux.NewRouter(),
	}
	s.setupRoutes()
	return s
}

func (s *Server) setupRoutes() {
	s.router.HandleFunc("/status", s.handleStatus).Methods("GET")
	s.router.HandleFunc("/trades", s.handleTrades).Methods("GET")
	s.router.HandleFunc("/metrics", s.handleMetrics).Methods("GET")
	s.router.HandleFunc("/start", s.handleStart).Methods("POST")
	s.router.HandleFunc("/stop", s.handleStop).Methods("POST")

	s.router.Use(corsMiddleware)
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *Server) ListenAndServe(port int) error {
	addr := ":" + strconv.Itoa(port)
	log.Info().Str("addr", addr).Msg("Starting HTTP server")
	return http.ListenAndServe(addr, s.router)
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	state := s.engine.GetState()
	riskState := s.engine.GetRiskManager().GetState()
	pos := s.engine.GetRiskManager().GetPosition()

	resp := map[string]interface{}{
		"is_running":   state.IsRunning,
		"last_update":  state.LastUpdate,
		"last_signal":  int(state.LastSignal),
		"error_count":  state.ErrorCount,
		"is_halted":    riskState.IsHalted,
		"halt_reason":  riskState.HaltReason,
		"equity":       riskState.CurrentEquity,
		"daily_pnl":    riskState.DailyPnL,
		"position": map[string]interface{}{
			"side":        pos.Side.String(),
			"entry_price": pos.EntryPrice,
			"entry_time":  pos.EntryTime,
			"size":        pos.Size,
		},
	}

	writeJSON(w, resp)
}

func (s *Server) handleTrades(w http.ResponseWriter, r *http.Request) {
	trades := s.engine.GetRiskManager().GetTrades()

	resp := make([]map[string]interface{}, len(trades))
	for i, t := range trades {
		resp[i] = map[string]interface{}{
			"id":          t.ID,
			"entry_time":  t.EntryTime,
			"exit_time":   t.ExitTime,
			"side":        t.Side.String(),
			"entry_price": t.EntryPrice,
			"exit_price":  t.ExitPrice,
			"size":        t.Size,
			"pnl":         t.PnL,
			"pnl_pct":     t.PnLPct,
			"exit_reason": t.ExitReason,
		}
	}

	writeJSON(w, resp)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := s.engine.GetRiskManager().GetMetrics()
	writeJSON(w, metrics)
}

func (s *Server) handleStart(w http.ResponseWriter, r *http.Request) {
	go s.engine.Start(r.Context())
	writeJSON(w, map[string]string{"status": "started"})
}

func (s *Server) handleStop(w http.ResponseWriter, r *http.Request) {
	s.engine.Stop()
	writeJSON(w, map[string]string{"status": "stopped"})
}

func writeJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}
