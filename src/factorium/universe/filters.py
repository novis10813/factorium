from __future__ import annotations

import polars as pl

from .rules import SymbolMetadata


class TagFilter:
    """Filter symbols by include/exclude tag sets."""

    def __init__(self, include: list[str] | None = None, exclude: list[str] | None = None) -> None:
        if include is None and exclude is None:
            raise ValueError("At least one of include or exclude must be specified")
        self._include = set(include) if include else None
        self._exclude = set(exclude) if exclude else None

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr:
        del df
        if tags is None:
            raise ValueError("TagFilter requires tags, but tags=None was provided")

        included_symbols: set[str] = set()
        for sym, meta in metadata.items():
            base_asset = meta.get("base_asset", "")
            symbol_tags = set(tags.get(base_asset, []))

            if self._include is not None and not (symbol_tags & self._include):
                continue

            if self._exclude is not None and (symbol_tags & self._exclude):
                continue

            included_symbols.add(sym)

        return pl.col("symbol").is_in(included_symbols)


class MinVolume:
    """Filter symbols by rolling mean traded volume."""

    def __init__(self, window: int = 20, threshold: float = 1_000_000) -> None:
        self._window = window
        self._threshold = threshold

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr:
        del df, metadata, tags
        rolling_volume = (
            pl.col("volume").rolling_mean(window_size=self._window, min_samples=self._window).over("symbol")
        )
        return (rolling_volume >= self._threshold).fill_null(False)


class MinLiquidity:
    """Filter symbols by rolling mean notional turnover."""

    def __init__(self, window: int = 20, threshold: float = 100_000) -> None:
        self._window = window
        self._threshold = threshold

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr:
        del df, metadata, tags
        liquidity = pl.col("volume") * pl.col("close")
        rolling_liquidity = liquidity.rolling_mean(window_size=self._window, min_samples=self._window).over("symbol")
        return (rolling_liquidity >= self._threshold).fill_null(False)
