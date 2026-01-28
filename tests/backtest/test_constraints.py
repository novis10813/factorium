import pytest
import polars as pl

from factorium.backtest.constraints import (
    MaxPositionConstraint,
    LongOnlyConstraint,
)


class TestMaxPositionConstraint:
    """Tests for MaxPositionConstraint."""

    def test_clips_weights_above_max(self):
        """Should clip weights exceeding max_weight."""
        weights = pl.DataFrame({
            "end_time": [1704067200000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.2, 0.05, -0.15],
        })
        
        constraint = MaxPositionConstraint(max_weight=0.1)
        result = constraint.apply(weights)
        
        assert result["weight"].to_list() == [0.1, 0.05, -0.1]

    def test_requires_positive_max_weight(self):
        """Should raise error for non-positive max_weight."""
        with pytest.raises(ValueError, match="must be positive"):
            MaxPositionConstraint(max_weight=0.0)


class TestLongOnlyConstraint:
    """Tests for LongOnlyConstraint."""

    def test_sets_negative_weights_to_zero(self):
        """Should set negative weights to zero."""
        weights = pl.DataFrame({
            "end_time": [1704067200000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.5, -0.3, 0.2],
        })
        
        constraint = LongOnlyConstraint()
        result = constraint.apply(weights)
        
        assert result["weight"].to_list() == [0.5, 0.0, 0.2]
