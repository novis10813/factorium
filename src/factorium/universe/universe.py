from __future__ import annotations

import polars as pl

from .rules import FilterRule, SymbolMetadata


class Universe:
    """Trading universe built from exclusion rules."""

    def __init__(self, rules: list[FilterRule]) -> None:
        if not rules:
            raise ValueError("Universe must have at least one rule")
        self.rules = rules

    def apply(
        self,
        df: pl.LazyFrame,
        metadata: dict[str, SymbolMetadata],
        tags: dict[str, list[str]] | None = None,
    ) -> pl.Expr:
        combined = self.rules[0].apply(df, metadata, tags)
        for rule in self.rules[1:]:
            combined = combined & rule.apply(df, metadata, tags)
        return combined
