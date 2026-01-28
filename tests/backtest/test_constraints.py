import pytest
import polars as pl

from factorium.backtest.constraints import (
    MaxPositionConstraint,
    LongOnlyConstraint,
    MaxGrossExposureConstraint,
    MarketNeutralConstraint,
)


class TestMaxPositionConstraint:
    """Tests for MaxPositionConstraint."""

    def test_clips_weights_above_max(self):
        """Should clip weights exceeding max_weight."""
        weights = pl.DataFrame(
            {
                "end_time": [1704067200000] * 3,
                "symbol": ["A", "B", "C"],
                "weight": [0.2, 0.05, -0.15],
            }
        )

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
        weights = pl.DataFrame(
            {
                "end_time": [1704067200000] * 3,
                "symbol": ["A", "B", "C"],
                "weight": [0.5, -0.3, 0.2],
            }
        )

        constraint = LongOnlyConstraint()
        result = constraint.apply(weights)

        assert result["weight"].to_list() == [0.5, 0.0, 0.2]


class TestMaxGrossExposureConstraint:
    """Tests for MaxGrossExposureConstraint."""

    def test_scales_weights_to_max_exposure(self):
        """Should scale weights proportionally to meet max_exposure."""
        weights = pl.DataFrame(
            {
                "end_time": [1000] * 3,
                "symbol": ["A", "B", "C"],
                "weight": [0.6, 0.5, -0.3],  # gross = 1.4
            }
        )

        constraint = MaxGrossExposureConstraint(max_exposure=1.0)
        result = constraint.apply(weights)

        gross = result["weight"].abs().sum()
        assert abs(gross - 1.0) < 1e-6

    def test_requires_positive_max_exposure(self):
        """Should raise error for non-positive max_exposure."""
        with pytest.raises(ValueError, match="must be positive"):
            MaxGrossExposureConstraint(max_exposure=0.0)


class TestMarketNeutralConstraint:
    """Tests for MarketNeutralConstraint."""

    def test_ensures_zero_sum_weights(self):
        """Should adjust weights to sum to zero."""
        weights = pl.DataFrame(
            {
                "end_time": [1000] * 2,
                "symbol": ["A", "B"],
                "weight": [0.6, 0.4],
            }
        )

        constraint = MarketNeutralConstraint()
        result = constraint.apply(weights)

        assert abs(result["weight"].sum()) < 1e-10
