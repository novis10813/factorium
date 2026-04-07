"""Weight allocators for the Alpha Pipeline.

Each allocator converts normalized signals to portfolio weights
satisfying specific invariants (e.g., market neutral, long-only).
"""

from abc import ABC, abstractmethod

import polars as pl

from ..constants import EPSILON


class WeightAllocator(ABC):
    """Convert normalized signal to portfolio weights."""

    @abstractmethod
    def allocate(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        """Add a 'weight' column satisfying the allocator's invariants.

        Args:
            df: DataFrame containing the signal column
            signal_col: Name of the normalized signal column
            group_col: Column to group by for cross-sectional operations

        Returns:
            DataFrame with 'weight' column added
        """
        ...

    @abstractmethod
    def renormalize(self, df: pl.DataFrame, group_col: str) -> pl.DataFrame:
        """Restore weight invariants after constraint application.

        Args:
            df: DataFrame with 'weight' column
            group_col: Column to group by

        Returns:
            DataFrame with renormalized weights
        """
        ...


class MarketNeutralAllocator(WeightAllocator):
    """Dollar-neutral allocator: sum(w)=0, sum(|w|)=1."""

    def allocate(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        demeaned = pl.col(signal_col) - pl.col(signal_col).mean().over(group_col)
        abs_sum = demeaned.abs().sum().over(group_col)
        weight = (demeaned / abs_sum).fill_nan(0.0).fill_null(0.0)
        return df.with_columns(weight.alias("weight"))

    def renormalize(self, df: pl.DataFrame, group_col: str) -> pl.DataFrame:
        df = df.with_columns(
            (pl.col("weight") - pl.col("weight").mean().over(group_col)).alias("weight")
        )
        abs_sum = pl.col("weight").abs().sum().over(group_col)
        return df.with_columns(
            pl.when(abs_sum > EPSILON)
            .then(pl.col("weight") / abs_sum)
            .otherwise(0.0)
            .alias("weight")
        )


class LongOnlyAllocator(WeightAllocator):
    """Long-only allocator: sum(w)=1, all w>=0. Only positive signals get weight."""

    def allocate(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        positive = (
            pl.when(pl.col(signal_col) > 0)
            .then(pl.col(signal_col))
            .otherwise(0.0)
        )
        w_sum = positive.sum().over(group_col)
        weight = (
            pl.when(w_sum > EPSILON)
            .then(positive / w_sum)
            .otherwise(0.0)
        )
        return df.with_columns(weight.fill_null(0.0).alias("weight"))

    def renormalize(self, df: pl.DataFrame, group_col: str) -> pl.DataFrame:
        df = df.with_columns(
            pl.when(pl.col("weight") < 0.0)
            .then(0.0)
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        w_sum = pl.col("weight").sum().over(group_col)
        return df.with_columns(
            pl.when(w_sum > EPSILON)
            .then(pl.col("weight") / w_sum)
            .otherwise(0.0)
            .alias("weight")
        )
