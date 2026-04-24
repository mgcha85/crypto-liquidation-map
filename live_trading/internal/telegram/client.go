package telegram

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/rs/zerolog/log"
)

type Client struct {
	botToken string
	chatID   string
	enabled  bool
	client   *http.Client
}

func New(botToken, chatID string, enabled bool) *Client {
	return &Client{
		botToken: botToken,
		chatID:   chatID,
		enabled:  enabled,
		client:   &http.Client{Timeout: 10 * time.Second},
	}
}

func (c *Client) SetEnabled(enabled bool) {
	c.enabled = enabled
}

func (c *Client) IsEnabled() bool {
	return c.enabled
}

func (c *Client) SendTradeNotification(side string, entryPrice, exitPrice, pnl, pnlPct float64) error {
	if !c.enabled || c.botToken == "" || c.chatID == "" {
		return nil
	}

	emoji := "🔴"
	if pnl > 0 {
		emoji = "🟢"
	}

	msg := fmt.Sprintf(
		"%s *Trade Closed*\n\n"+
			"Side: %s\n"+
			"Entry: $%.2f\n"+
			"Exit: $%.2f\n"+
			"PnL: $%.2f (%.2f%%)",
		emoji, side, entryPrice, exitPrice, pnl, pnlPct*100,
	)

	return c.send(msg)
}

func (c *Client) SendAlert(message string) error {
	if !c.enabled || c.botToken == "" || c.chatID == "" {
		return nil
	}
	return c.send("⚠️ " + message)
}

func (c *Client) send(text string) error {
	url := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", c.botToken)

	payload := map[string]interface{}{
		"chat_id":    c.chatID,
		"text":       text,
		"parse_mode": "Markdown",
	}

	body, _ := json.Marshal(payload)
	resp, err := c.client.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		log.Error().Err(err).Msg("Failed to send Telegram message")
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Warn().Int("status", resp.StatusCode).Msg("Telegram API error")
	}

	return nil
}
