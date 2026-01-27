"""Polars-based computation engine for high-performance factor operations."""

import polars as pl
import pandas as pd
import numpy as np

from ..constants import EPSILON


class PolarsEngine:
    """High-performance factor computation engine using Polars.

    Provides efficient implementations of:
    - Time-series operations (rolling mean, std, shift, diff)
    - Cross-sectional operations (rank, zscore)
    - Data format conversion (Polars <-> Pandas)

    All operations are designed to work with the standard factor DataFrame format:
    - start_time: int64 (milliseconds)
    - end_time: int64 (milliseconds)
    - symbol: str
    - factor: float64
    """

    @staticmethod
    def ts_mean(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling mean with strict window (NaN if window not full).

        Args:
            df: Input DataFrame
            window: Rolling window size
            symbol_col: Column name for symbol grouping
            value_col: Column name for values

        Returns:
            DataFrame with rolling mean applied
        """
        return df.with_columns(
            pl.col(value_col).rolling_mean(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )

    @staticmethod
    def ts_std(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling standard deviation with strict window."""
        return df.with_columns(
            pl.col(value_col).rolling_std(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )

    @staticmethod
    def ts_sum(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling sum with strict window."""
        return df.with_columns(
            pl.col(value_col).rolling_sum(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )

    @staticmethod
    def ts_min(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling minimum with strict window."""
        return df.with_columns(
            pl.col(value_col).rolling_min(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )

    @staticmethod
    def ts_max(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling maximum with strict window."""
        return df.with_columns(
            pl.col(value_col).rolling_max(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )

    @staticmethod
    def ts_shift(
        df: pl.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Shift values by period within each symbol."""
        return df.with_columns(pl.col(value_col).shift(period).over(symbol_col).alias(value_col))

    @staticmethod
    def ts_diff(
        df: pl.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Difference from period ago within each symbol."""
        return df.with_columns(pl.col(value_col).diff(period).over(symbol_col).alias(value_col))

    @staticmethod
    def cs_rank(
        df: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional percentile rank (0 to 1). Strict: Returns NaN if any input is NaN."""
        # Check if any value in the group is NaN, if so return NaN for all
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        rank_expr = (
            pl.col(value_col)
            .rank(method="min", descending=False)
            .over(time_col)
            .truediv(pl.col(value_col).count().over(time_col))
        )
        return df.with_columns(pl.when(has_nan).then(None).otherwise(rank_expr).alias(value_col))

    @staticmethod
    def cs_zscore(
        df: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional z-score standardization. Strict: Returns NaN if any input is NaN."""
        # Check if any value in the group is NaN, if so return NaN for all
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        zscore_expr = (pl.col(value_col) - pl.col(value_col).mean().over(time_col)) / pl.col(value_col).std().over(
            time_col
        )
        return df.with_columns(pl.when(has_nan).then(None).otherwise(zscore_expr).alias(value_col))

    @staticmethod
    def cs_demean(
        df: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional de-meaning. Strict: Returns NaN if any input is NaN."""
        # Check if any value in the group is NaN, if so return NaN for all
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        demean_expr = pl.col(value_col) - pl.col(value_col).mean().over(time_col)
        return df.with_columns(pl.when(has_nan).then(None).otherwise(demean_expr).alias(value_col))

    @staticmethod
    def cs_winsorize(
        df: pl.DataFrame,
        limits: float = 0.025,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional winsorization. Strict: Returns NaN if any input is NaN."""
        # Check if any value in the group is NaN, if so return NaN for all
        has_nan = pl.col(value_col).is_null().any().over(time_col)

        # Calculate quantile bounds
        lower_bound = pl.col(value_col).quantile(limits).over(time_col)
        upper_bound = pl.col(value_col).quantile(1 - limits).over(time_col)

        # Clip values between bounds
        winsorize_expr = pl.col(value_col).clip(min_bound=lower_bound, max_bound=upper_bound)

        return df.with_columns(pl.when(has_nan).then(None).otherwise(winsorize_expr).alias(value_col))

    @staticmethod
    def cs_mean(
        df: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional mean. Strict: Returns NaN if any input is NaN."""
        # Check if any value in the group is NaN, if so return NaN for all
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        mean_expr = pl.col(value_col).mean().over(time_col)
        return df.with_columns(pl.when(has_nan).then(None).otherwise(mean_expr).alias(value_col))

    @staticmethod
    def cs_median(
        df: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional median. Strict: Returns NaN if any input is NaN."""
        # Check if any value in the group is NaN, if so return NaN for all
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        median_expr = pl.col(value_col).median().over(time_col)
        return df.with_columns(pl.when(has_nan).then(None).otherwise(median_expr).alias(value_col))

    @staticmethod
    def cs_neutralize(
        df_y: pl.DataFrame,
        df_x: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional neutralization via least-squares regression.

        Returns residuals of: Y = alpha + beta * X + residuals
        Strict: Returns NaN if any input is NaN in the cross-section.

        Uses Polars map_batches with least-squares regression.
        """
        # Merge Y and X data on time and symbol
        df = df_y.join(
            df_x.select(["end_time", "symbol", value_col]).rename({value_col: "factor_x"}),
            on=["end_time", "symbol"],
            how="inner",
        ).rename({value_col: "factor_y"})

        # Group by end_time and compute residuals via map_batches
        def least_squares_batch(batch_df):
            # batch_df is a Polars DataFrame for one time period
            batch_pd = batch_df.to_pandas()

            # Check for NaNs in either factor - if any, return all NaN
            if batch_pd[["factor_y", "factor_x"]].isna().any().any():
                batch_pd["residual"] = np.nan
                return batch_pd

            y = batch_pd["factor_y"].values.astype(float)
            x = batch_pd["factor_x"].values.astype(float)

            # Check for constant x (cannot regress)
            if np.std(x) < EPSILON:
                batch_pd["residual"] = np.nan
                return batch_pd

            try:
                # Solve least squares: y = [x 1] * [beta alpha]^T
                A = np.vstack([x, np.ones(len(x))]).T
                beta_alpha, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                residuals = y - A @ beta_alpha
                batch_pd["residual"] = residuals
            except Exception:
                batch_pd["residual"] = np.nan

            return batch_pd

        # Apply least-squares per time period
        result = df.map_batches(least_squares_batch, schema={**df.schema, "residual": pl.Float64})

        # Select and rename back to factor
        result = result.select(["start_time", "end_time", "symbol", "residual"]).rename({"residual": value_col})

        return result

    @staticmethod
    def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
        """Convert Polars DataFrame to Pandas."""
        return df.to_pandas()

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> pl.DataFrame:
        """Convert Pandas DataFrame to Polars."""
        return pl.from_pandas(df)
