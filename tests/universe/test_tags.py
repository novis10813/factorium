import json
from pathlib import Path

import pytest

import factorium.data.loader as data_loader
from factorium.universe.tags import TagProvider


def test_fetch_maps_symbols_to_categories(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = TagProvider(cache_dir=tmp_path, cache_ttl=0)

    async def fake_request_json(session, url, params=None, headers=None):
        del session, params, headers
        if url.endswith("/coins/list"):
            return [
                {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
                {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
            ]
        if url.endswith("/coins/bitcoin"):
            return {"categories": ["Layer 1", "Store Of Value"]}
        if url.endswith("/coins/ethereum"):
            return {"categories": ["Layer 1", "Smart Contract Platform"]}
        raise AssertionError(f"unexpected url: {url}")

    async def no_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr(provider, "_request_json", fake_request_json)
    monkeypatch.setattr("factorium.universe.tags.asyncio.sleep", no_sleep)

    out = provider.fetch(symbols=["BTC", "ETH"])
    assert out["BTC"] == ["Layer 1", "Store Of Value"]
    assert out["ETH"] == ["Layer 1", "Smart Contract Platform"]


def test_fetch_prefers_canonical_id_on_symbol_collision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = TagProvider(cache_dir=tmp_path, cache_ttl=0)

    async def fake_request_json(session, url, params=None, headers=None):
        del session, params, headers
        if url.endswith("/coins/list"):
            return [
                {"id": "btc-token", "symbol": "btc", "name": "Some BTC Token"},
                {"id": "btc", "symbol": "btc", "name": "Canonical BTC"},
            ]
        if url.endswith("/coins/btc"):
            return {"categories": ["Store Of Value"]}
        if url.endswith("/coins/btc-token"):
            raise AssertionError("should use canonical id when collision exists")
        raise AssertionError(f"unexpected url: {url}")

    async def no_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr(provider, "_request_json", fake_request_json)
    monkeypatch.setattr("factorium.universe.tags.asyncio.sleep", no_sleep)

    out = provider.fetch(symbols=["BTC"])
    assert out["BTC"] == ["Store Of Value"]


def test_cache_load_save_and_ttl(tmp_path: Path) -> None:
    provider = TagProvider(cache_dir=tmp_path, cache_ttl=10)
    sample = {"BTC": ["Layer 1"], "ETH": ["Layer 1", "Smart Contract Platform"]}

    provider._save_cache(sample)
    loaded = provider._load_cache()
    assert loaded == sample

    cache_path = tmp_path / "coingecko_tags.json"
    blob = json.loads(cache_path.read_text(encoding="utf-8"))
    blob["saved_at"] = 0
    cache_path.write_text(json.dumps(blob), encoding="utf-8")

    assert provider._load_cache() is None


def test_fetch_uses_cached_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = TagProvider(cache_dir=tmp_path, cache_ttl=3600)
    provider._save_cache({"BTC": ["Layer 1"], "ETH": ["Layer 1"]})

    async def should_not_call(*args, **kwargs):
        raise AssertionError("network should not be called for cached subset")

    monkeypatch.setattr(provider, "_request_json", should_not_call)

    out = provider.fetch(symbols=["BTC"])
    assert out == {"BTC": ["Layer 1"]}


def test_tags_fetch_uses_run_async(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = TagProvider()
    expected = {"BTC": ["Layer 1"]}

    async def fake_fetch_async(symbols=None):
        del symbols
        return expected

    called = {"value": False}

    def fake_run_async(coro):
        called["value"] = True
        coro.close()
        return expected

    monkeypatch.setattr(provider, "fetch_async", fake_fetch_async)
    monkeypatch.setattr(data_loader, "_run_async", fake_run_async)

    out = provider.fetch(symbols=["BTC"])
    assert out == expected
    assert called["value"] is True


def test_fetch_raises_when_symbols_is_none(tmp_path: Path) -> None:
    provider = TagProvider(cache_dir=tmp_path, cache_ttl=0)
    with pytest.raises(ValueError, match="symbols must be provided"):
        provider.fetch(symbols=None)
