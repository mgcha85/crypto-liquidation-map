package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

// XGBParams - LOCKED from Optuna Trial #24
type XGBParams struct {
	MaxDepth       int     `yaml:"max_depth"`
	LearningRate   float64 `yaml:"learning_rate"`
	NEstimators    int     `yaml:"n_estimators"`
	Subsample      float64 `yaml:"subsample"`
	ColsampleBytree float64 `yaml:"colsample_bytree"`
	Gamma          float64 `yaml:"gamma"`
	RegAlpha       float64 `yaml:"reg_alpha"`
	RegLambda      float64 `yaml:"reg_lambda"`
}

// BarrierConfig - Triple barrier parameters
type BarrierConfig struct {
	ProfitTake   float64 `yaml:"profit_take"`
	StopLoss     float64 `yaml:"stop_loss"`
	HorizonHours int     `yaml:"horizon_hours"`
}

// PositionConfig - Position sizing
type PositionConfig struct {
	SizePct     float64 `yaml:"size_pct"`
	TakerFeePct float64 `yaml:"taker_fee_pct"`
	SlippageBps float64 `yaml:"slippage_bps"`
	Leverage    int     `yaml:"leverage"`
}

// RiskConfig - Risk limits
type RiskConfig struct {
	DailyLossLimitPct  float64 `yaml:"daily_loss_limit_pct"`
	WeeklyLossLimitPct float64 `yaml:"weekly_loss_limit_pct"`
	MaxPositions       int     `yaml:"max_positions"`
}

// APIConfig - Binance API settings
type APIConfig struct {
	Key        string `yaml:"key"`
	Secret     string `yaml:"secret"`
	Testnet    bool   `yaml:"testnet"`
	TimeoutSec int    `yaml:"timeout_sec"`
}

// Config - Main configuration
type Config struct {
	Mode              string         `yaml:"mode"`
	Symbol            string         `yaml:"symbol"`
	LookbackHours     int            `yaml:"lookback_hours"`
	UpdateIntervalSec int            `yaml:"update_interval_sec"`
	ModelPath         string         `yaml:"model_path"`
	LogLevel          string         `yaml:"log_level"`
	HTTPPort          int            `yaml:"http_port"`

	TelegramBotToken string `yaml:"telegram_bot_token"`
	TelegramChatID   string `yaml:"telegram_chat_id"`
	TelegramEnabled  bool   `yaml:"telegram_enabled"`

	Barrier  BarrierConfig  `yaml:"barrier"`
	Position PositionConfig `yaml:"position"`
	Risk     RiskConfig     `yaml:"risk"`
	API      APIConfig      `yaml:"api"`
}

// DefaultConfig returns SOTA-locked configuration
func DefaultConfig() *Config {
	return &Config{
		Mode:              "paper",
		Symbol:            "BTCUSDT",
		LookbackHours:     50,
		UpdateIntervalSec: 3600,
		ModelPath:         "models/xgb_optuna_best.onnx",
		LogLevel:          "info",
		HTTPPort:          8080,
		Barrier: BarrierConfig{
			ProfitTake:   0.02,
			StopLoss:     0.01,
			HorizonHours: 48,
		},
		Position: PositionConfig{
			SizePct:     0.10,
			TakerFeePct: 0.0004,
			SlippageBps: 5.0,
			Leverage:    1,
		},
		Risk: RiskConfig{
			DailyLossLimitPct:  0.02,
			WeeklyLossLimitPct: 0.05,
			MaxPositions:       1,
		},
		API: APIConfig{
			Testnet:    true,
			TimeoutSec: 30,
		},
	}
}

// Load reads configuration from YAML file
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	cfg := DefaultConfig()
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	if cfg.API.Key == "" {
		cfg.API.Key = os.Getenv("BINANCE_API_KEY")
	}
	if cfg.API.Secret == "" {
		cfg.API.Secret = os.Getenv("BINANCE_API_SECRET")
	}
	if cfg.TelegramBotToken == "" {
		cfg.TelegramBotToken = os.Getenv("TELEGRAM_BOT_TOKEN")
	}
	if cfg.TelegramChatID == "" {
		cfg.TelegramChatID = os.Getenv("TELEGRAM_CHAT_ID")
	}

	return cfg, nil
}

// FeatureNames - 31 features matching backtest
var FeatureNames = []string{
	"total_intensity", "long_intensity", "short_intensity", "long_short_ratio",
	"above_below_ratio", "near_1pct_concentration", "near_2pct_concentration",
	"near_5pct_concentration", "largest_long_cluster_distance", "largest_short_cluster_distance",
	"largest_long_cluster_volume", "largest_short_cluster_volume",
	"top3_long_dist_1", "top3_long_dist_2", "top3_long_dist_3",
	"top3_short_dist_1", "top3_short_dist_2", "top3_short_dist_3",
	"entropy", "skewness",
	"return_1h", "return_6h", "return_12h", "return_24h",
	"volatility_6h", "volatility_24h", "atr_24h", "volume_ma_ratio",
	"wick_ratio_upper", "wick_ratio_lower", "price_position",
}
