"""Signal normalizers for the Alpha Pipeline.

Each normalizer transforms raw alpha signals to a known range
before weight allocation.
"""

from abc import ABC, abstractmethod

import polars as pl

from ..constants import EPSILON


class Normalizer(ABC):
    """Transform raw alpha to a known range."""

    @abstractmethod
    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        """Normalize signal_col in-place (overwrite the column).

        Args:
            df: DataFrame containing the signal column
            signal_col: Name of the signal column to normalize
            group_col: Column to group by for cross-sectional operations

        Returns:
            DataFrame with signal_col replaced by normalized values
        """
        ...


class RawNormalizer(Normalizer):
    """Pass-through normalizer. No transformation applied."""

    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        return df


class RankNormalizer(Normalizer):
    """Cross-sectional rank normalization to [0, 1]."""

    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        rank = pl.col(signal_col).rank().over(group_col)
        count = pl.col(signal_col).count().over(group_col)
        return df.with_columns((rank / count).alias(signal_col))


class ZScoreNormalizer(Normalizer):
    """Cross-sectional z-score normalization to approximately [-3, 3]."""

    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        mean = pl.col(signal_col).mean().over(group_col)
        std = pl.col(signal_col).std().over(group_col)
        return df.with_columns(
            ((pl.col(signal_col) - mean) / std)
            .fill_nan(None)
            .alias(signal_col)
        )


class MinMaxNormalizer(Normalizer):
    """Cross-sectional min-max normalization to [0, 1]."""

    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        min_val = pl.col(signal_col).min().over(group_col)
        max_val = pl.col(signal_col).max().over(group_col)
        denom = max_val - min_val
        signal = pl.col(signal_col)
        return df.with_columns(
            pl.when(denom.abs() <= EPSILON)
            .then(pl.lit(None))
            .otherwise((signal - min_val) / denom)
            .alias(signal_col)
        )
