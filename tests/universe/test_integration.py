import polars as pl

from factorium import AggBar
from factorium.factors.analyzer import FactorAnalyzer
from factorium.universe import (
    Checklist,
    ExcludeStablecoins,
    MinListingAge,
    MinVolume,
    TagFilter,
    Universe,
)


DAY_MS = 86_400_000
BASE_TS = 1_700_000_000_000


def _make_aggbar() -> AggBar:
    rows = []
    for i in range(10):
        ts = BASE_TS + i * DAY_MS
        rows.extend(
            [
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "BTCUSDT",
                    "open": 100 + i,
                    "high": 101 + i,
                    "low": 99 + i,
                    "close": 100 + i,
                    "volume": 20_000 + i,
                    "alpha": float(i + 1),
                },
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "USDCUSDT",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 50_000,
                    "alpha": float(50 - i),
                },
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "NEWUSDT",
                    "open": 10 + i,
                    "high": 11 + i,
                    "low": 9 + i,
                    "close": 10 + i,
                    "volume": 100 + i,
                    "alpha": float(100 + i),
                },
            ]
        )
    return AggBar(pl.DataFrame(rows))


def _metadata() -> dict[str, dict]:
    return {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "status": "TRADING",
            "listing_date": BASE_TS - 365 * DAY_MS,
        },
        "USDCUSDT": {
            "symbol": "USDCUSDT",
            "base_asset": "USDC",
            "quote_asset": "USDT",
            "status": "TRADING",
            "listing_date": BASE_TS - 365 * DAY_MS,
            "is_stablecoin_pair": True,
        },
        "NEWUSDT": {
            "symbol": "NEWUSDT",
            "base_asset": "NEW",
            "quote_asset": "USDT",
            "status": "TRADING",
            "listing_date": BASE_TS - 2 * DAY_MS,
        },
    }


def test_full_pipeline_universe_checklist_factor_eval() -> None:
    bar = _make_aggbar()
    metadata = _metadata()
    tags = {"BTC": ["layer1"], "USDC": ["stablecoin"], "NEW": ["meme"]}

    universe = Universe([ExcludeStablecoins(), MinListingAge(days=5)])
    checklist = Checklist([TagFilter(include=["layer1"]), MinVolume(window=3, threshold=10_000)])

    bar = bar.with_mask("in_universe", universe, metadata)
    bar = bar.with_mask("in_checklist", checklist, metadata, tags)

    factor = bar["alpha"]
    result = factor.eval(prices=bar, periods=1, quantiles=2, mask="in_checklist")

    analyzer = FactorAnalyzer(factor=factor, prices=bar, quantiles=2, mask="in_checklist")
    prepared = analyzer.prepare_data(periods=[1], price_col="close")

    assert set(prepared["symbol"].to_list()) == {"BTCUSDT"}
    assert result.factor_name == "alpha"
