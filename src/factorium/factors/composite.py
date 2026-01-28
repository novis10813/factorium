"""
Multi-factor composition support.

Allows combining multiple factors using weighted combinations.
"""

from typing import List, Dict, Optional
import polars as pl

from .core import Factor


class CompositeFactor:
    """
    Weighted combination of multiple factors.

    Args:
        factors: List of Factor objects
        weights: Optional weights (defaults to equal weight)
        name: Name for the composite factor

    Example:
        >>> momentum = data["close"].ts_return(20)
        >>> value = data["volume"].cs_rank()
        >>> composite = CompositeFactor([momentum, value], weights=[0.6, 0.4])
        >>> signal = composite.to_factor()
    """

    def __init__(
        self,
        factors: List[Factor],
        weights: Optional[List[float]] = None,
        name: str = "composite",
    ):
        if len(factors) == 0:
            raise ValueError("At least one factor required")

        if weights is None:
            weights = [1.0 / len(factors)] * len(factors)

        if len(weights) != len(factors):
            raise ValueError("Number of weights must match number of factors")

        self.factors = factors
        self.weights = weights
        self.name = name

    @classmethod
    def from_equal_weights(cls, factors: List[Factor], name: str = "composite") -> "CompositeFactor":
        """
        Create composite with equal weights.

        Args:
            factors: List of factors
            name: Name for composite

        Returns:
            CompositeFactor with equal weights
        """
        return cls(factors, weights=None, name=name)

    @classmethod
    def from_weights(cls, factors: List[Factor], weights: List[float], name: str = "composite") -> "CompositeFactor":
        """
        Create composite with custom weights.

        Args:
            factors: List of factors
            weights: Custom weights (must sum to 1)
            name: Name for composite

        Returns:
            CompositeFactor with given weights
        """
        # Validate weights sum to approximately 1
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        return cls(factors, weights=weights, name=name)

    @classmethod
    def from_zscore(cls, factors: List[Factor], name: str = "composite_zscore") -> "CompositeFactor":
        """
        Create composite by standardizing factors first (z-score).

        Each factor is standardized: (x - mean) / std
        Then combined with equal weights.

        Args:
            factors: List of factors
            name: Name for composite

        Returns:
            CompositeFactor with z-score normalized factors
        """
        import numpy as np

        # Standardize each factor
        standardized = []
        for factor in factors:
            df = factor.lazy.collect()

            # Calculate mean and std per symbol (cross-sectionally)
            df = df.with_columns(
                [
                    pl.col("factor").mean().over(["start_time", "end_time"]).alias("cs_mean"),
                    pl.col("factor").std().over(["start_time", "end_time"]).alias("cs_std"),
                ]
            )

            # Z-score: (factor - mean) / std
            df = df.with_columns(
                [
                    ((pl.col("factor") - pl.col("cs_mean")) / pl.col("cs_std"))
                    .fill_nan(0.0)  # Handle division by zero
                    .alias("factor")
                ]
            )

            df = df.select(["start_time", "end_time", "symbol", "factor"])

            from .core import Factor

            standardized.append(Factor(df, name=f"{factor.name}_zscore"))

        return cls(standardized, weights=None, name=name)

    def to_factor(self) -> Factor:
        """
        Combine factors into a single Factor.

        Returns:
            Factor representing weighted combination
        """
        # Start with first factor
        result = self.factors[0].lazy.with_columns((pl.col("factor") * self.weights[0]).alias("weighted"))

        # Add remaining factors
        for i, factor in enumerate(self.factors[1:], start=1):
            factor_df = factor.lazy.with_columns((pl.col("factor") * self.weights[i]).alias(f"weighted_{i}"))
            result = result.join(
                factor_df.select(["start_time", "end_time", "symbol", f"weighted_{i}"]),
                on=["start_time", "end_time", "symbol"],
                how="left",
            )
            result = result.with_columns((pl.col("weighted") + pl.col(f"weighted_{i}").fill_null(0)).alias("weighted"))

        # Collect and create Factor
        result_df = result.select(["start_time", "end_time", "symbol", "weighted"]).collect()
        result_df = result_df.rename({"weighted": "factor"})

        return Factor(result_df, name=self.name)
