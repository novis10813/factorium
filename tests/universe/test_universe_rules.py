import polars as pl

from factorium.universe import ExcludeLeveragedTokens, ExcludeStablecoins, MinListingAge, Universe


NOW_MS = 1_700_000_000_000
DAY_MS = 86_400_000


def _sample_df() -> pl.DataFrame:
    rows = [
        {"start_time": NOW_MS - 5 * DAY_MS, "end_time": NOW_MS - 5 * DAY_MS + 3_600_000, "symbol": "BTCUSDT"},
        {"start_time": NOW_MS - 5 * DAY_MS, "end_time": NOW_MS - 5 * DAY_MS + 3_600_000, "symbol": "USDCUSDT"},
        {"start_time": NOW_MS - 5 * DAY_MS, "end_time": NOW_MS - 5 * DAY_MS + 3_600_000, "symbol": "BTCUPUSDT"},
        {"start_time": NOW_MS - 5 * DAY_MS, "end_time": NOW_MS - 5 * DAY_MS + 3_600_000, "symbol": "NEWUSDT"},
        {"start_time": NOW_MS + 100 * DAY_MS, "end_time": NOW_MS + 100 * DAY_MS + 3_600_000, "symbol": "NEWUSDT"},
    ]
    return pl.DataFrame(rows)


def _sample_metadata() -> dict[str, dict]:
    return {
        "BTCUSDT": {
            "symbol": "BTCUSDT",
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "status": "TRADING",
            "listing_date": NOW_MS - 365 * DAY_MS,
        },
        "USDCUSDT": {
            "symbol": "USDCUSDT",
            "base_asset": "USDC",
            "quote_asset": "USDT",
            "status": "TRADING",
            "is_stablecoin_pair": True,
            "listing_date": NOW_MS - 365 * DAY_MS,
        },
        "BTCUPUSDT": {
            "symbol": "BTCUPUSDT",
            "base_asset": "BTCUP",
            "quote_asset": "USDT",
            "status": "TRADING",
            "is_leveraged": True,
            "listing_date": NOW_MS - 365 * DAY_MS,
        },
        "NEWUSDT": {
            "symbol": "NEWUSDT",
            "base_asset": "NEW",
            "quote_asset": "USDT",
            "status": "TRADING",
            "listing_date": NOW_MS - 10 * DAY_MS,
        },
    }


def test_exclude_stablecoins_filters_stable_base_assets() -> None:
    df = _sample_df().lazy()
    metadata = _sample_metadata()
    expr = ExcludeStablecoins().apply(df, metadata)
    out = _sample_df().lazy().with_columns(expr.alias("keep")).collect()

    stable_keep = out.filter(pl.col("symbol") == "USDCUSDT")["keep"].to_list()
    assert stable_keep == [False]


def test_exclude_leveraged_tokens_filters_leveraged_symbols() -> None:
    df = _sample_df().lazy()
    metadata = _sample_metadata()
    expr = ExcludeLeveragedTokens().apply(df, metadata)
    out = _sample_df().lazy().with_columns(expr.alias("keep")).collect()

    leveraged_keep = out.filter(pl.col("symbol") == "BTCUPUSDT")["keep"].to_list()
    assert leveraged_keep == [False]


def test_min_listing_age_is_time_varying() -> None:
    metadata = _sample_metadata()
    df = _sample_df().lazy()
    expr = MinListingAge(days=90).apply(df, metadata)
    out = _sample_df().lazy().with_columns(expr.alias("keep")).collect()

    new_rows = out.filter(pl.col("symbol") == "NEWUSDT").sort("start_time")
    assert new_rows["keep"].to_list() == [False, True]


def test_universe_combines_rules_with_and_logic() -> None:
    metadata = _sample_metadata()
    rules = [ExcludeStablecoins(), ExcludeLeveragedTokens(), MinListingAge(days=90)]
    universe = Universe(rules)

    out = _sample_df().lazy().with_columns(universe.apply(_sample_df().lazy(), metadata).alias("in_universe")).collect()
    kept_symbols = set(out.filter(pl.col("in_universe"))["symbol"].to_list())
    assert kept_symbols == {"BTCUSDT", "NEWUSDT"}


def test_min_listing_age_excludes_symbol_when_listing_date_missing() -> None:
    metadata = _sample_metadata()
    metadata["NEWUSDT"].pop("listing_date")

    out = (
        _sample_df()
        .lazy()
        .with_columns(MinListingAge(days=90).apply(_sample_df().lazy(), metadata).alias("keep"))
        .collect()
    )
    new_rows = out.filter(pl.col("symbol") == "NEWUSDT").sort("start_time")
    assert new_rows["keep"].to_list() == [False, False]
