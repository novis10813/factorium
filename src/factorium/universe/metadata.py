from __future__ import annotations

import json
import time
from pathlib import Path

import aiohttp

from ..data.loader import _run_async
from .rules import KNOWN_STABLECOINS, LEVERAGED_PATTERNS, SymbolMetadata


ENDPOINTS = {
    "spot": "https://api.binance.com/api/v3/exchangeInfo",
    "um": "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "cm": "https://dapi.binance.com/dapi/v1/exchangeInfo",
}


class MetadataProvider:
    """Fetch and cache symbol metadata from Binance exchangeInfo."""

    def __init__(self, market: str = "um", cache_dir: str | Path = "./Data/metadata", cache_ttl: int = 86400) -> None:
        if market not in ENDPOINTS:
            raise ValueError(f"Unsupported market: {market}")
        self.market = market
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl
        self._cache_path = self.cache_dir / f"{self.market}_exchange_info.json"

    async def _request_json(self, session: aiohttp.ClientSession, url: str) -> dict:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def fetch_async(self) -> dict[str, SymbolMetadata]:
        cached = self._load_cache()
        if cached is not None:
            return cached

        endpoint = ENDPOINTS[self.market]
        async with aiohttp.ClientSession() as session:
            payload = await self._request_json(session, endpoint)

        parsed = self._parse_exchange_info(payload)
        self._save_cache(parsed)
        return parsed

    def fetch(self) -> dict[str, SymbolMetadata]:
        return _run_async(self.fetch_async())

    def _parse_exchange_info(self, data: dict) -> dict[str, SymbolMetadata]:
        output: dict[str, SymbolMetadata] = {}
        for item in data.get("symbols", []):
            symbol = item.get("symbol")
            if not symbol:
                continue

            base_asset = item.get("baseAsset", "")
            onboard_date = item.get("onboardDate")
            listing_date = int(onboard_date) if isinstance(onboard_date, (int, float)) else None

            output[symbol] = {
                "symbol": symbol,
                "base_asset": base_asset,
                "quote_asset": item.get("quoteAsset", ""),
                "status": item.get("status", ""),
                "listing_date": listing_date,
                "is_leveraged": bool(LEVERAGED_PATTERNS.search(base_asset)),
                "is_stablecoin_pair": base_asset in KNOWN_STABLECOINS,
            }

        return output

    def _load_cache(self) -> dict[str, SymbolMetadata] | None:
        if not self._cache_path.exists():
            return None

        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        saved_at = payload.get("saved_at")
        if not isinstance(saved_at, (int, float)):
            return None

        if time.time() - float(saved_at) > self.cache_ttl:
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        return data

    def _save_cache(self, data: dict[str, SymbolMetadata]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"saved_at": time.time(), "data": data}
        self._cache_path.write_text(json.dumps(payload), encoding="utf-8")
