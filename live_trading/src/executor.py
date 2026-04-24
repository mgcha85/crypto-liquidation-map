import asyncio
import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

import aiohttp
import polars as pl


@dataclass
class BinanceConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    timeout: int = 30
    
    @property
    def base_url(self) -> str:
        if self.testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"


class BinanceExecutor:
    
    def __init__(self, config: BinanceConfig | None = None):
        self.config = config or BinanceConfig(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
        )
        self._session: aiohttp.ClientSession | None = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session
    
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _sign(self, params: dict) -> str:
        query_string = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        signed: bool = False,
    ) -> dict:
        session = await self._get_session()
        url = f"{self.config.base_url}{endpoint}"
        
        params = params or {}
        headers = {"X-MBX-APIKEY": self.config.api_key}
        
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)
        
        async with session.request(method, url, params=params, headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise Exception(f"API error {resp.status}: {data}")
            return data
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 50,
    ) -> pl.DataFrame:
        data = await self._request(
            "GET",
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        
        return pl.DataFrame({
            "open_time": [int(k[0]) for k in data],
            "open": [float(k[1]) for k in data],
            "high": [float(k[2]) for k in data],
            "low": [float(k[3]) for k in data],
            "close": [float(k[4]) for k in data],
            "volume": [float(k[5]) for k in data],
            "close_time": [int(k[6]) for k in data],
        })
    
    async def get_open_interest_hist(
        self,
        symbol: str,
        period: str = "1h",
        limit: int = 50,
    ) -> pl.DataFrame:
        data = await self._request(
            "GET",
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        
        return pl.DataFrame({
            "timestamp": [int(d["timestamp"]) for d in data],
            "sumOpenInterest": [float(d["sumOpenInterest"]) for d in data],
            "sumOpenInterestValue": [float(d["sumOpenInterestValue"]) for d in data],
        })
    
    async def get_account(self) -> dict:
        return await self._request("GET", "/fapi/v2/account", signed=True)
    
    async def get_position_risk(self, symbol: str | None = None) -> list[dict]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = await self._request("GET", "/fapi/v2/positionRisk", params, signed=True)
        return data
    
    async def place_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        order_type: str = "MARKET",
        reduce_only: bool = False,
    ) -> dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": f"{quantity:.6f}",
        }
        
        if reduce_only:
            params["reduceOnly"] = "true"
        
        return await self._request("POST", "/fapi/v1/order", params, signed=True)
    
    async def close_position(self, symbol: str) -> dict | None:
        positions = await self.get_position_risk(symbol)
        
        for pos in positions:
            if pos["symbol"] == symbol:
                amt = float(pos["positionAmt"])
                if amt != 0:
                    side = "SELL" if amt > 0 else "BUY"
                    return await self.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=abs(amt),
                        reduce_only=True,
                    )
        return None


def create_data_fetcher(executor: BinanceExecutor):
    async def fetch(symbol: str, lookback_hours: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        df_klines, df_oi = await asyncio.gather(
            executor.get_klines(symbol, "1h", lookback_hours),
            executor.get_open_interest_hist(symbol, "1h", lookback_hours),
        )
        return df_oi, df_klines
    
    def sync_fetch(symbol: str, lookback_hours: int) -> tuple[pl.DataFrame, pl.DataFrame]:
        return asyncio.run(fetch(symbol, lookback_hours))
    
    return sync_fetch
