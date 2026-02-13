from __future__ import annotations

import polars as pl

from .rules import FilterRule, SymbolMetadata


class Checklist:
    """Research checklist built from inclusion filters."""

    def __init__(self, filters: list[FilterRule]) -> None:
        if not filters:
            raise ValueError("Checklist must have at least one filter")
        self.filters = filters

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr:
        combined = self.filters[0].apply(df, metadata, tags)
        for flt in self.filters[1:]:
            combined = combined & flt.apply(df, metadata, tags)
        return combined
