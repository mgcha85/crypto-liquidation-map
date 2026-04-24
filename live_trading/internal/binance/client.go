package binance

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/mgcha85/crypto-liquidation-map/live_trading/internal/config"
)

type Client struct {
	cfg        *config.APIConfig
	httpClient *http.Client
	baseURL    string
}

func NewClient(cfg *config.APIConfig) *Client {
	baseURL := "https://fapi.binance.com"
	if cfg.Testnet {
		baseURL = "https://testnet.binancefuture.com"
	}

	return &Client{
		cfg:     cfg,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Duration(cfg.TimeoutSec) * time.Second,
		},
	}
}

func (c *Client) sign(params url.Values) string {
	mac := hmac.New(sha256.New, []byte(c.cfg.Secret))
	mac.Write([]byte(params.Encode()))
	return hex.EncodeToString(mac.Sum(nil))
}

func (c *Client) request(method, endpoint string, params url.Values, signed bool) ([]byte, error) {
	if params == nil {
		params = url.Values{}
	}

	if signed {
		params.Set("timestamp", strconv.FormatInt(time.Now().UnixMilli(), 10))
		params.Set("signature", c.sign(params))
	}

	reqURL := c.baseURL + endpoint
	if len(params) > 0 && method == "GET" {
		reqURL += "?" + params.Encode()
	}

	req, err := http.NewRequest(method, reqURL, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("X-MBX-APIKEY", c.cfg.Key)

	if method == "POST" && len(params) > 0 {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	return body, nil
}

type Kline struct {
	OpenTime  int64
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	CloseTime int64
}

func (c *Client) GetKlines(symbol, interval string, limit int) ([]Kline, error) {
	params := url.Values{
		"symbol":   {symbol},
		"interval": {interval},
		"limit":    {strconv.Itoa(limit)},
	}

	body, err := c.request("GET", "/fapi/v1/klines", params, false)
	if err != nil {
		return nil, err
	}

	var raw [][]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}

	klines := make([]Kline, len(raw))
	for i, k := range raw {
		klines[i] = Kline{
			OpenTime:  int64(k[0].(float64)),
			Open:      parseFloat(k[1]),
			High:      parseFloat(k[2]),
			Low:       parseFloat(k[3]),
			Close:     parseFloat(k[4]),
			Volume:    parseFloat(k[5]),
			CloseTime: int64(k[6].(float64)),
		}
	}

	return klines, nil
}

type OpenInterest struct {
	Timestamp int64
	OI        float64
	OIValue   float64
}

func (c *Client) GetOpenInterestHist(symbol, period string, limit int) ([]OpenInterest, error) {
	params := url.Values{
		"symbol": {symbol},
		"period": {period},
		"limit":  {strconv.Itoa(limit)},
	}

	body, err := c.request("GET", "/futures/data/openInterestHist", params, false)
	if err != nil {
		return nil, err
	}

	var raw []map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}

	ois := make([]OpenInterest, len(raw))
	for i, o := range raw {
		ois[i] = OpenInterest{
			Timestamp: int64(o["timestamp"].(float64)),
			OI:        parseFloat(o["sumOpenInterest"]),
			OIValue:   parseFloat(o["sumOpenInterestValue"]),
		}
	}

	return ois, nil
}

func (c *Client) GetAccount() (map[string]interface{}, error) {
	body, err := c.request("GET", "/fapi/v2/account", nil, true)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return result, nil
}

type PositionRisk struct {
	Symbol       string
	PositionAmt  float64
	EntryPrice   float64
	UnrealizedPL float64
	Leverage     int
}

func (c *Client) GetPositionRisk(symbol string) ([]PositionRisk, error) {
	params := url.Values{}
	if symbol != "" {
		params.Set("symbol", symbol)
	}

	body, err := c.request("GET", "/fapi/v2/positionRisk", params, true)
	if err != nil {
		return nil, err
	}

	var raw []map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}

	positions := make([]PositionRisk, 0)
	for _, p := range raw {
		amt := parseFloat(p["positionAmt"])
		if amt != 0 || symbol != "" {
			positions = append(positions, PositionRisk{
				Symbol:       p["symbol"].(string),
				PositionAmt:  amt,
				EntryPrice:   parseFloat(p["entryPrice"]),
				UnrealizedPL: parseFloat(p["unRealizedProfit"]),
				Leverage:     int(parseFloat(p["leverage"])),
			})
		}
	}

	return positions, nil
}

type OrderResult struct {
	OrderID   int64
	Symbol    string
	Side      string
	Status    string
	Price     float64
	AvgPrice  float64
	Qty       float64
	Timestamp int64
}

func (c *Client) PlaceOrder(symbol, side string, quantity float64, reduceOnly bool) (*OrderResult, error) {
	params := url.Values{
		"symbol":   {symbol},
		"side":     {side},
		"type":     {"MARKET"},
		"quantity": {fmt.Sprintf("%.6f", quantity)},
	}

	if reduceOnly {
		params.Set("reduceOnly", "true")
	}

	body, err := c.request("POST", "/fapi/v1/order", params, true)
	if err != nil {
		return nil, err
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}

	return &OrderResult{
		OrderID:   int64(raw["orderId"].(float64)),
		Symbol:    raw["symbol"].(string),
		Side:      raw["side"].(string),
		Status:    raw["status"].(string),
		Price:     parseFloat(raw["price"]),
		AvgPrice:  parseFloat(raw["avgPrice"]),
		Qty:       parseFloat(raw["executedQty"]),
		Timestamp: int64(raw["updateTime"].(float64)),
	}, nil
}

func (c *Client) ClosePosition(symbol string) (*OrderResult, error) {
	positions, err := c.GetPositionRisk(symbol)
	if err != nil {
		return nil, err
	}

	for _, p := range positions {
		if p.Symbol == symbol && p.PositionAmt != 0 {
			side := "SELL"
			qty := p.PositionAmt
			if qty < 0 {
				side = "BUY"
				qty = -qty
			}
			return c.PlaceOrder(symbol, side, qty, true)
		}
	}

	return nil, nil
}

func parseFloat(v interface{}) float64 {
	switch val := v.(type) {
	case float64:
		return val
	case string:
		f, _ := strconv.ParseFloat(val, 64)
		return f
	default:
		return 0
	}
}
