import polars as pl
import pytest

from factorium import AggBar
from factorium.universe import (
    Checklist,
    ExcludeStablecoins,
    MinListingAge,
    TagFilter,
    Universe,
)


DAY_MS = 86_400_000
BASE_TS = 1_700_000_000_000


class SymbolOnlyRule:
    def __init__(self, allowed: set[str]) -> None:
        self.allowed = allowed

    def apply(self, df: pl.LazyFrame, metadata: dict, tags: dict[str, list[str]] | None = None) -> pl.Expr:
        del df, metadata, tags
        return pl.col("symbol").is_in(self.allowed)


def _sample_aggbar() -> AggBar:
    rows = []
    for i in range(2):
        ts = BASE_TS + i * DAY_MS
        for symbol, close in [("BTCUSDT", 100.0), ("USDCUSDT", 1.0), ("NEWUSDT", 10.0)]:
            rows.append(
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": symbol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1_000.0,
                }
            )
    return AggBar(pl.DataFrame(rows))


def _metadata() -> dict[str, dict]:
    return {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "listing_date": BASE_TS - 500 * DAY_MS,
        },
        "USDCUSDT": {
            "symbol": "USDCUSDT",
            "base_asset": "USDC",
            "quote_asset": "USDT",
            "is_stablecoin_pair": True,
            "listing_date": BASE_TS - 500 * DAY_MS,
        },
        "NEWUSDT": {"symbol": "NEWUSDT", "base_asset": "NEW", "quote_asset": "USDT", "listing_date": BASE_TS - DAY_MS},
    }


def test_with_mask_returns_new_aggbar_without_mutating_original() -> None:
    agg = _sample_aggbar()
    masked = agg.with_mask("in_universe", SymbolOnlyRule({"BTCUSDT"}), _metadata())

    assert "in_universe" not in agg.cols
    assert "in_universe" in masked.cols
    values = masked.to_polars().filter(pl.col("symbol") == "BTCUSDT")["in_universe"].to_list()
    assert values == [True, True]


def test_with_mask_rejects_protected_column_name() -> None:
    agg = _sample_aggbar()
    with pytest.raises(ValueError, match="protected column name"):
        agg.with_mask("close", SymbolOnlyRule({"BTCUSDT"}), _metadata())


def test_getitem_factor_does_not_include_mask_column() -> None:
    agg = _sample_aggbar().with_mask("in_universe", SymbolOnlyRule({"BTCUSDT"}), _metadata())

    factor = agg["close"]
    assert factor.lazy.collect().columns == ["start_time", "end_time", "symbol", "factor"]


def test_universe_and_checklist_masks_can_be_composed() -> None:
    agg = _sample_aggbar()
    metadata = _metadata()
    tags = {"BTC": ["layer1"], "USDC": ["stablecoin"], "NEW": ["meme"]}

    universe = Universe([ExcludeStablecoins(), MinListingAge(days=2)])
    checklist = Checklist([TagFilter(include=["layer1"])])

    with_universe = agg.with_mask("in_universe", universe, metadata)
    with_checklist = with_universe.with_mask("in_checklist", checklist, metadata, tags)

    df = with_checklist.to_polars()
    kept = set(df.filter(pl.col("in_universe") & pl.col("in_checklist"))["symbol"].to_list())
    assert kept == {"BTCUSDT"}


def test_with_mask_supports_time_varying_rule() -> None:
    agg = _sample_aggbar()
    metadata = _metadata()
    masked = agg.with_mask("old_enough", Universe([MinListingAge(days=2)]), metadata)

    new_values = masked.to_polars().filter(pl.col("symbol") == "NEWUSDT").sort("start_time")["old_enough"].to_list()
    assert new_values == [False, True]
