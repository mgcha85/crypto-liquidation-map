package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"

	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/api"
	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/config"
	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/engine"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	configPath := flag.String("config", "configs/paper.yaml", "Config file path")
	capital := flag.Float64("capital", 10000.0, "Initial capital in USDT")
	flag.Parse()

	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load config")
	}

	switch cfg.LogLevel {
	case "debug":
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	case "warn":
		zerolog.SetGlobalLevel(zerolog.WarnLevel)
	case "error":
		zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	default:
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}

	eng, err := engine.New(cfg, *capital)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create engine")
	}
	defer eng.Close()

	server := api.NewServer(eng)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := server.ListenAndServe(cfg.HTTPPort); err != nil {
			log.Fatal().Err(err).Msg("HTTP server failed")
		}
	}()

	go eng.Start(ctx)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Info().Msg("Shutting down...")
	eng.Stop()
}
