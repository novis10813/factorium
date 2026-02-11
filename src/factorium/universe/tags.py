from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import aiohttp


COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


class TagProvider:
    """Fetch and cache token categories from CoinGecko."""

    def __init__(
        self,
        cache_dir: str | Path = "./Data/metadata",
        cache_ttl: int = 604800,
        api_key: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl
        self.api_key = api_key
        self._cache_path = self.cache_dir / "coingecko_tags.json"

    async def _request_json(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict | list:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    async def fetch_async(self, symbols: list[str] | None = None) -> dict[str, list[str]]:
        requested = [s.upper() for s in symbols] if symbols is not None else None
        cached = self._load_cache()

        if cached is not None:
            if requested is None:
                return cached
            if all(sym in cached for sym in requested):
                return {sym: cached[sym] for sym in requested}

        headers: dict[str, str] | None = None
        if self.api_key:
            headers = {"x-cg-pro-api-key": self.api_key}

        async with aiohttp.ClientSession() as session:
            raw_list = await self._request_json(session, f"{COINGECKO_BASE_URL}/coins/list", headers=headers)
            symbol_to_id: dict[str, str] = {}
            for item in raw_list if isinstance(raw_list, list) else []:
                symbol = str(item.get("symbol", "")).upper()
                coin_id = item.get("id")
                if symbol and coin_id and symbol not in symbol_to_id:
                    symbol_to_id[symbol] = str(coin_id)

            targets = requested or sorted(symbol_to_id.keys())
            result: dict[str, list[str]] = {} if cached is None else dict(cached)

            for symbol in targets:
                if symbol in result:
                    continue

                coin_id = symbol_to_id.get(symbol)
                if not coin_id:
                    continue

                detail = await self._request_json(session, f"{COINGECKO_BASE_URL}/coins/{coin_id}", headers=headers)
                categories = detail.get("categories", []) if isinstance(detail, dict) else []
                result[symbol] = [str(tag) for tag in categories]
                await asyncio.sleep(0.12)

        self._save_cache(result)
        if requested is None:
            return result
        return {sym: result.get(sym, []) for sym in requested if sym in result}

    def fetch(self, symbols: list[str] | None = None) -> dict[str, list[str]]:
        return asyncio.run(self.fetch_async(symbols=symbols))

    def _load_cache(self) -> dict[str, list[str]] | None:
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

        out: dict[str, list[str]] = {}
        for key, value in data.items():
            if isinstance(value, list):
                out[str(key)] = [str(v) for v in value]
        return out

    def _save_cache(self, data: dict[str, list[str]]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"saved_at": time.time(), "data": data}
        self._cache_path.write_text(json.dumps(payload), encoding="utf-8")
