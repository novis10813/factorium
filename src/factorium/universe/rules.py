from __future__ import annotations

import re
from typing import Protocol, TypedDict, runtime_checkable

import polars as pl


class SymbolMetadata(TypedDict, total=False):
    """Metadata for a single trading symbol."""

    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    listing_date: int
    is_leveraged: bool
    is_stablecoin_pair: bool


@runtime_checkable
class FilterRule(Protocol):
    """Shared interface for Universe rules and Checklist filters."""

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr: ...


KNOWN_STABLECOINS = frozenset(
    {
        "USDT",
        "USDC",
        "BUSD",
        "DAI",
        "TUSD",
        "FDUSD",
        "USDP",
        "USDD",
        "UST",
        "FRAX",
        "LUSD",
        "SUSD",
        "GUSD",
        "USDJ",
        "EUR",
        "GBP",
        "AEUR",
    }
)


LEVERAGED_PATTERNS = re.compile(r"(UP|DOWN|BEAR|BULL|[0-9]+[LS])$", re.IGNORECASE)


class ExcludeStablecoins:
    """Exclude symbols where base asset is a stablecoin."""

    def __init__(self, extra_stablecoins: set[str] | None = None) -> None:
        self._extra = extra_stablecoins or set()

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr:
        del df, tags
        all_stables = KNOWN_STABLECOINS | self._extra
        excluded_symbols = [
            sym
            for sym, meta in metadata.items()
            if meta.get("base_asset", "") in all_stables or bool(meta.get("is_stablecoin_pair", False))
        ]
        return ~pl.col("symbol").is_in(excluded_symbols)


class ExcludeLeveragedTokens:
    """Exclude leveraged-token symbols."""

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr:
        del df, tags
        excluded_symbols = [
            sym
            for sym, meta in metadata.items()
            if bool(meta.get("is_leveraged", False)) or bool(LEVERAGED_PATTERNS.search(meta.get("base_asset", "")))
        ]
        return ~pl.col("symbol").is_in(excluded_symbols)


class MinListingAge:
    """Exclude symbols younger than configured listing days."""

    def __init__(self, days: int = 90) -> None:
        self._min_ms = days * 86_400_000

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr:
        del df, tags
        listing_map: dict[str, int] = {}
        for sym, meta in metadata.items():
            listing_date = meta.get("listing_date")
            if listing_date is not None:
                listing_map[sym] = int(listing_date)

        if not listing_map:
            return pl.lit(True)

        listing_expr = pl.col("symbol").replace_strict(listing_map, default=None).cast(pl.Int64, strict=False)
        return ((pl.col("start_time") - listing_expr) >= self._min_ms) | listing_expr.is_null()
