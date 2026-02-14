import json
from pathlib import Path

import pytest

import factorium.data.loader as data_loader
from factorium.universe.metadata import MetadataProvider


def test_parse_exchange_info_extracts_symbol_metadata() -> None:
    provider = MetadataProvider(market="um")
    payload = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1_700_000_000_000,
            },
            {
                "symbol": "BTCUPUSDT",
                "baseAsset": "BTCUP",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "onboardDate": 1_700_000_000_000,
            },
            {
                "symbol": "USDCUSDT",
                "baseAsset": "USDC",
                "quoteAsset": "USDT",
                "status": "TRADING",
            },
        ]
    }

    out = provider._parse_exchange_info(payload)

    assert out["BTCUSDT"]["listing_date"] == 1_700_000_000_000
    assert out["BTCUSDT"]["is_leveraged"] is False
    assert out["BTCUSDT"]["is_stablecoin_pair"] is False
    assert out["BTCUPUSDT"]["is_leveraged"] is True
    assert out["USDCUSDT"]["is_stablecoin_pair"] is True


def test_cache_load_save_and_ttl(tmp_path: Path) -> None:
    provider = MetadataProvider(market="um", cache_dir=tmp_path, cache_ttl=60)
    sample = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "status": "TRADING",
            "listing_date": 1_700_000_000_000,
            "is_leveraged": False,
            "is_stablecoin_pair": False,
        }
    }

    provider._save_cache(sample)
    loaded = provider._load_cache()
    assert loaded == sample

    cache_path = tmp_path / "um_exchange_info.json"
    blob = json.loads(cache_path.read_text(encoding="utf-8"))
    blob["saved_at"] = 0
    cache_path.write_text(json.dumps(blob), encoding="utf-8")

    assert provider._load_cache() is None


def test_fetch_prefers_cache_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MetadataProvider(market="um", cache_dir=tmp_path, cache_ttl=3600)
    cached = {
        "ETHUSDT": {
            "symbol": "ETHUSDT",
            "base_asset": "ETH",
            "quote_asset": "USDT",
            "status": "TRADING",
            "listing_date": 1_700_000_000_000,
            "is_leveraged": False,
            "is_stablecoin_pair": False,
        }
    }
    provider._save_cache(cached)

    async def should_not_call(*args, **kwargs):
        raise AssertionError("network should not be called when cache is valid")

    monkeypatch.setattr(provider, "_request_json", should_not_call)

    out = provider.fetch()
    assert out == cached


def test_metadata_fetch_uses_run_async(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MetadataProvider(market="um")
    expected = {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "status": "TRADING",
            "listing_date": 1_700_000_000_000,
            "is_leveraged": False,
            "is_stablecoin_pair": False,
        }
    }

    async def fake_fetch_async() -> dict[str, dict[str, object]]:
        return expected

    called = {"value": False}

    def fake_run_async(coro):
        called["value"] = True
        coro.close()
        return expected

    monkeypatch.setattr(provider, "fetch_async", fake_fetch_async)
    monkeypatch.setattr(data_loader, "_run_async", fake_run_async)

    out = provider.fetch()
    assert out == expected
    assert called["value"] is True
