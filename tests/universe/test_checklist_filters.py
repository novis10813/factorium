import polars as pl
import pytest

from factorium.universe import Checklist, MinLiquidity, MinVolume, TagFilter


def _make_panel() -> pl.DataFrame:
    rows = []
    base_ts = 1_700_000_000_000
    for i in range(5):
        ts = base_ts + i * 3_600_000
        rows.extend(
            [
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "BTCUSDT",
                    "close": 100.0,
                    "volume": 20_000.0,
                },
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "DOGEUSDT",
                    "close": 1.0,
                    "volume": 100.0,
                },
            ]
        )
    return pl.DataFrame(rows)


def _metadata() -> dict[str, dict]:
    return {
        "BTCUSDT": {"symbol": "BTCUSDT", "base_asset": "BTC", "quote_asset": "USDT"},
        "DOGEUSDT": {"symbol": "DOGEUSDT", "base_asset": "DOGE", "quote_asset": "USDT"},
    }


def _tags() -> dict[str, list[str]]:
    return {
        "BTC": ["store-of-value", "layer1"],
        "DOGE": ["meme"],
    }


def test_tag_filter_include_mode() -> None:
    df = _make_panel().lazy()
    expr = TagFilter(include=["store-of-value"]).apply(df, _metadata(), _tags())
    out = _make_panel().lazy().with_columns(expr.alias("keep")).collect()
    assert set(out.filter(pl.col("keep"))["symbol"].to_list()) == {"BTCUSDT"}


def test_tag_filter_exclude_mode() -> None:
    df = _make_panel().lazy()
    expr = TagFilter(exclude=["meme"]).apply(df, _metadata(), _tags())
    out = _make_panel().lazy().with_columns(expr.alias("keep")).collect()
    assert set(out.filter(pl.col("keep"))["symbol"].to_list()) == {"BTCUSDT"}


def test_tag_filter_raises_when_tags_missing() -> None:
    with pytest.raises(ValueError, match="requires tags"):
        TagFilter(include=["layer1"]).apply(_make_panel().lazy(), _metadata(), tags=None)


def test_min_volume_uses_rolling_threshold() -> None:
    df = _make_panel().lazy()
    expr = MinVolume(window=3, threshold=10_000).apply(df, _metadata())
    out = _make_panel().lazy().with_columns(expr.alias("keep")).collect()

    btc = out.filter(pl.col("symbol") == "BTCUSDT").sort("start_time")["keep"].to_list()
    doge = out.filter(pl.col("symbol") == "DOGEUSDT").sort("start_time")["keep"].to_list()
    assert btc == [False, False, True, True, True]
    assert doge == [False, False, False, False, False]


def test_min_liquidity_uses_volume_times_close() -> None:
    df = _make_panel().lazy()
    expr = MinLiquidity(window=3, threshold=500_000).apply(df, _metadata())
    out = _make_panel().lazy().with_columns(expr.alias("keep")).collect()

    btc = out.filter(pl.col("symbol") == "BTCUSDT").sort("start_time")["keep"].to_list()
    doge = out.filter(pl.col("symbol") == "DOGEUSDT").sort("start_time")["keep"].to_list()
    assert btc == [False, False, True, True, True]
    assert doge == [False, False, False, False, False]


def test_checklist_combines_filters_with_and_logic() -> None:
    checklist = Checklist([TagFilter(exclude=["meme"]), MinVolume(window=3, threshold=10_000)])
    out = (
        _make_panel()
        .lazy()
        .with_columns(checklist.apply(_make_panel().lazy(), _metadata(), _tags()).alias("ok"))
        .collect()
    )
    symbols = set(out.filter(pl.col("ok"))["symbol"].to_list())
    assert symbols == {"BTCUSDT"}
