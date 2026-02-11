"""
Weight constraints for backtesting.

Provides constraints for position sizing and weight bounds.
"""

from typing import Optional, Dict
import polars as pl
from abc import ABC, abstractmethod


class WeightConstraint(ABC):
    """
    Base class for weight constraints.

    Constraints modify weights before position calculation to enforce
    limits like max position size, sector exposure, etc.
    """

    @abstractmethod
    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        """
        Apply constraint to weights.

        Args:
            weights: DataFrame with columns: end_time, symbol, weight

        Returns:
            Modified DataFrame with constrained weights
        """
        pass


class MaxPositionConstraint(WeightConstraint):
    """
    Limit maximum absolute weight per position.

    Args:
        max_weight: Maximum absolute weight (e.g., 0.1 for 10%)

    Example:
        >>> constraint = MaxPositionConstraint(max_weight=0.15)
        >>> constrained = constraint.apply(weights)
    """

    def __init__(self, max_weight: float):
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        self.max_weight = max_weight

    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        """Clip weights to [-max_weight, max_weight]."""
        return weights.with_columns(
            pl.when(pl.col("weight") > self.max_weight)
            .then(pl.lit(self.max_weight))
            .when(pl.col("weight") < -self.max_weight)
            .then(pl.lit(-self.max_weight))
            .otherwise(pl.col("weight"))
            .alias("weight")
        )


class LongOnlyConstraint(WeightConstraint):
    """
    Force all weights to be non-negative (long-only).

    Negative weights are set to zero.
    """

    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        """Set negative weights to zero."""
        return weights.with_columns(
            pl.when(pl.col("weight") < 0).then(pl.lit(0.0)).otherwise(pl.col("weight")).alias("weight")
        )


class MaxGrossExposureConstraint(WeightConstraint):
    """Limit sum(|weights|) per timestamp."""

    def __init__(self, max_exposure: float):
        if max_exposure <= 0:
            raise ValueError("max_exposure must be positive")
        self.max_exposure = max_exposure

    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        # Group by end_time, calculate gross, scale if needed
        gross = weights.group_by("end_time").agg(pl.col("weight").abs().sum().alias("gross"))

        weights = weights.join(gross, on="end_time")

        weights = weights.with_columns(
            pl.when(pl.col("gross") > self.max_exposure)
            .then(pl.col("weight") * self.max_exposure / pl.col("gross"))
            .otherwise(pl.col("weight"))
            .alias("weight")
        )

        return weights.drop("gross")


class MarketNeutralConstraint(WeightConstraint):
    """Enforce sum(weights) = 0 per timestamp."""

    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        # Calculate mean per timestamp
        means = weights.group_by("end_time").agg(pl.col("weight").mean().alias("mean_w"))

        weights = weights.join(means, on="end_time")

        weights = weights.with_columns((pl.col("weight") - pl.col("mean_w")).alias("weight"))

        return weights.drop("mean_w")
