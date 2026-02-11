import polars as pl
import pytest

from factorium.universe import Checklist, FilterRule, Universe


class DummyRule:
    def apply(self, df: pl.LazyFrame, metadata: dict, tags: dict[str, list[str]] | None = None) -> pl.Expr:
        return pl.lit(True)


def test_filter_rule_runtime_protocol() -> None:
    rule = DummyRule()
    assert isinstance(rule, FilterRule)


def test_universe_requires_at_least_one_rule() -> None:
    with pytest.raises(ValueError, match="at least one rule"):
        Universe([])


def test_checklist_requires_at_least_one_filter() -> None:
    with pytest.raises(ValueError, match="at least one filter"):
        Checklist([])
