try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import cauchy, norm, uniform

from ...constants import EPSILON


class TimeSeriesOpsMixin:
    """Time-series operations mixin using pure Polars LazyFrame."""

    def ts_sum(self, window: int) -> Self:
        """Strict rolling sum: NaN if window contains NaN or insufficient length."""
        self._validate_window(window)

        # Add a row index to preserve original order
        result_lf = (
            self._lf.with_row_index("__row_idx__")
            .sort(["symbol", "end_time"])
            .with_columns(
                pl.col("factor").rolling_sum(window_size=window, min_samples=window).over("symbol").alias("factor")
            )
            .sort("__row_idx__")
            .drop("__row_idx__")
        )

        return self.__class__(result_lf, f"ts_sum({self.name},{window})")

    def ts_product(self, window: int) -> Self:
        """Strict rolling product: NaN if window contains NaN or insufficient length.

        Uses log/exp transformation for numerical stability:
        prod(x) = sign_prod * exp(sum(log(|x|)))
        Handles zero and negative values correctly.
        """
        self._validate_window(window)

        # Create NaN-in-window mask
        nan_in_window = (
            (pl.col("factor").is_null() | pl.col("factor").is_nan())
            .cast(pl.Int64)
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .fill_null(1)
        )

        # Check for zero in window (product = 0)
        has_zero = (
            (pl.col("factor") == 0)
            .cast(pl.Int64)
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .fill_null(0)
        )

        # Count negative values in window for sign determination
        neg_count = (
            (pl.col("factor") < 0).cast(pl.Int64).rolling_sum(window_size=window, min_samples=window).over("symbol")
        )

        # Product of absolute values via log/exp transformation
        abs_prod = pl.col("factor").abs().log().rolling_sum(window_size=window, min_samples=window).over("symbol").exp()

        # Determine sign: (-1)^neg_count
        sign_expr = pl.when(neg_count % 2 == 1).then(pl.lit(-1.0)).otherwise(pl.lit(1.0))

        # Final product: handle NaN, zero, and normal cases
        prod_expr = (
            pl.when(nan_in_window > 0)
            .then(pl.lit(None))
            .when(has_zero > 0)
            .then(pl.lit(0.0))
            .otherwise(sign_expr * abs_prod)
        )

        result_lf = (
            self._lf.sort(["symbol", "end_time"]).with_columns(prod_expr.alias("factor")).sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_product({self.name},{window})")

    def ts_mean(self, window: int) -> Self:
        """Strict rolling mean: NaN if window contains NaN or insufficient length."""
        self._validate_window(window)

        result_lf = (
            self._lf.sort(["symbol", "end_time"])
            .with_columns(
                pl.col("factor").rolling_mean(window_size=window, min_samples=window).over("symbol").alias("factor")
            )
            .sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_mean({self.name},{window})")

    def ts_median(self, window: int) -> Self:
        """Strict rolling median: NaN if window contains NaN or insufficient length."""
        self._validate_window(window)

        result_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.col("factor").rolling_median(window_size=window, min_samples=window).over("symbol").alias("factor")
        )

        return self.__class__(result_lf, f"ts_median({self.name},{window})")

    def ts_std(self, window: int) -> Self:
        """Strict rolling std: NaN if window contains NaN or insufficient length."""
        self._validate_window(window)

        result_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.col("factor").rolling_std(window_size=window, min_samples=window).over("symbol").alias("factor")
        )

        return self.__class__(result_lf, f"ts_std({self.name},{window})")

    def ts_min(self, window: int) -> Self:
        """Strict rolling min: NaN if window contains NaN or insufficient length."""
        self._validate_window(window)

        result_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.col("factor").rolling_min(window_size=window, min_samples=window).over("symbol").alias("factor")
        )

        return self.__class__(result_lf, f"ts_min({self.name},{window})")

    def ts_max(self, window: int) -> Self:
        """Strict rolling max: NaN if window contains NaN or insufficient length."""
        self._validate_window(window)

        result_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.col("factor").rolling_max(window_size=window, min_samples=window).over("symbol").alias("factor")
        )

        return self.__class__(result_lf, f"ts_max({self.name},{window})")

    def ts_argmin(self, window: int) -> Self:
        """Rolling argmin: position of minimum in window (pure Polars).

        Returns the distance from the end of the window to the minimum value position.
        NaN if window contains NaN or has insufficient length.
        """
        self._validate_window(window)

        # Add row index within each symbol group for rolling()
        sorted_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.int_range(pl.len()).over("symbol").alias("_idx")
        )

        # Create rolling windows as lists
        with_windows = sorted_lf.with_columns(
            pl.col("factor")
            .rolling(index_column="_idx", period=f"{window}i", closed="right")
            .over("symbol")
            .alias("_window")
        )

        # Calculate argmin with NaN checking
        with_argmin = with_windows.with_columns(
            [
                pl.col("_window").list.len().alias("_len"),
                # Check for NaN or null in window
                pl.col("_window")
                .list.eval(pl.element().is_nan().any() | pl.element().is_null().any())
                .list.first()
                .alias("_has_nan"),
                pl.col("_window").list.arg_min().alias("_argmin_raw"),
            ]
        )

        # Final result: (len - 1 - argmin) gives distance from end
        result_lf = (
            with_argmin.with_columns(
                pl.when(pl.col("_has_nan") | (pl.col("_len") < window))
                .then(None)
                .otherwise((pl.col("_len") - 1 - pl.col("_argmin_raw")).cast(pl.Float64))
                .alias("factor")
            )
            .drop(["_idx", "_window", "_len", "_has_nan", "_argmin_raw"])
            .sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_argmin({self.name},{window})")

    def ts_argmax(self, window: int) -> Self:
        """Rolling argmax: position of maximum in window (pure Polars).

        Returns the distance from the end of the window to the maximum value position.
        NaN if window contains NaN or has insufficient length.
        """
        self._validate_window(window)

        # Add row index within each symbol group for rolling()
        sorted_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.int_range(pl.len()).over("symbol").alias("_idx")
        )

        # Create rolling windows as lists
        with_windows = sorted_lf.with_columns(
            pl.col("factor")
            .rolling(index_column="_idx", period=f"{window}i", closed="right")
            .over("symbol")
            .alias("_window")
        )

        # Calculate argmax with NaN checking
        with_argmax = with_windows.with_columns(
            [
                pl.col("_window").list.len().alias("_len"),
                # Check for NaN or null in window
                pl.col("_window")
                .list.eval(pl.element().is_nan().any() | pl.element().is_null().any())
                .list.first()
                .alias("_has_nan"),
                pl.col("_window").list.arg_max().alias("_argmax_raw"),
            ]
        )

        # Final result: (len - 1 - argmax) gives distance from end
        result_lf = (
            with_argmax.with_columns(
                pl.when(pl.col("_has_nan") | (pl.col("_len") < window))
                .then(None)
                .otherwise((pl.col("_len") - 1 - pl.col("_argmax_raw")).cast(pl.Float64))
                .alias("factor")
            )
            .drop(["_idx", "_window", "_len", "_has_nan", "_argmax_raw"])
            .sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_argmax({self.name},{window})")

    def ts_scale(self, window: int, constant: float = 0) -> Self:
        self._validate_window(window)

        nan_in_window = (
            (pl.col("factor").is_null() | pl.col("factor").is_nan())
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .cast(pl.Int64)
            .fill_null(1)
        )

        min_expr = pl.col("factor").rolling_min(window_size=window, min_samples=window).over("symbol")
        max_expr = pl.col("factor").rolling_max(window_size=window, min_samples=window).over("symbol")
        denom = max_expr - min_expr
        scale_expr = (pl.col("factor") - min_expr) / denom
        scale_expr = pl.when(nan_in_window > 0).then(pl.lit(None)).otherwise(scale_expr)
        scale_expr = pl.when(denom.abs() <= EPSILON).then(pl.lit(None)).otherwise(scale_expr)
        scale_expr = pl.when(scale_expr.is_finite()).then(scale_expr).otherwise(pl.lit(None))

        result_lf = self._lf.with_columns((scale_expr + pl.lit(constant)).alias("factor"))
        return self.__class__(result_lf, f"ts_scale({self.name},{window},{constant})")

    def ts_zscore(self, window: int) -> Self:
        self._validate_window(window)

        nan_in_window = (
            (pl.col("factor").is_null() | pl.col("factor").is_nan())
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .cast(pl.Int64)
            .fill_null(1)
        )

        mean_expr = pl.col("factor").rolling_mean(window_size=window, min_samples=window).over("symbol")
        std_expr = pl.col("factor").rolling_std(window_size=window, min_samples=window).over("symbol")
        z_expr = (pl.col("factor") - mean_expr) / std_expr
        z_expr = pl.when(nan_in_window > 0).then(pl.lit(None)).otherwise(z_expr)
        z_expr = pl.when(std_expr.abs() <= EPSILON).then(pl.lit(None)).otherwise(z_expr)
        z_expr = pl.when(z_expr.is_finite()).then(z_expr).otherwise(pl.lit(None))

        result_lf = self._lf.with_columns(z_expr.alias("factor"))
        return self.__class__(result_lf, f"ts_zscore({self.name},{window})")

    def ts_quantile(self, window: int, driver: str = "gaussian") -> Self:
        """Transform percentile rank to quantile using PPF function (pure Polars)."""
        self._validate_window(window)

        valid_drivers = {
            "gaussian": norm.ppf,
            "uniform": uniform.ppf,
            "cauchy": cauchy.ppf,
        }
        if driver not in valid_drivers:
            raise ValueError(f"Invalid driver: {driver}. Valid drivers are: {list(valid_drivers.keys())}")

        ppf_func = valid_drivers[driver]
        ranked_factor = self.ts_rank(window)
        epsilon = 1e-6

        def apply_ppf(x):
            """Apply PPF to ranked values."""
            x = np.asarray(x)
            if np.isnan(x):
                return np.nan
            clipped = np.clip(x, epsilon, 1 - epsilon)
            return ppf_func(clipped)

        result_lf = ranked_factor._lf.with_columns(
            pl.col("factor").map_batches(lambda s: s.map_elements(apply_ppf, return_dtype=pl.Float64)).alias("factor")
        )

        return self.__class__(result_lf, f"ts_quantile({self.name},{window},{driver})")

    def ts_kurtosis(self, window: int) -> Self:
        """Rolling kurtosis with strict NaN semantics (pure Polars)."""
        self._validate_window(window)

        # Polars rolling_kurtosis handles NaN propagation natively
        result_lf = (
            self._lf.sort(["symbol", "end_time"])
            .with_columns(
                pl.col("factor").rolling_kurtosis(window_size=window, min_samples=window).over("symbol").alias("factor")
            )
            .sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_kurtosis({self.name},{window})")

    def ts_skewness(self, window: int) -> Self:
        """Rolling skewness with strict NaN semantics (pure Polars)."""
        self._validate_window(window)

        # Polars rolling_skew handles NaN propagation natively
        result_lf = (
            self._lf.sort(["symbol", "end_time"])
            .with_columns(
                pl.col("factor").rolling_skew(window_size=window, min_samples=window).over("symbol").alias("factor")
            )
            .sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_skewness({self.name},{window})")

    def ts_step(self, start: int = 1) -> Self:
        result_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.col("symbol").cumcount().over("symbol").add(start).alias("factor")
        )
        return self.__class__(result_lf, f"ts_step({self.name},{start})")

    def ts_shift(self, period: int) -> Self:
        """Shift values by period within each symbol group."""
        result_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.col("factor").shift(period).over("symbol").alias("factor")
        )

        return self.__class__(result_lf, f"ts_shift({self.name},{period})")

    def ts_delta(self, period: int) -> Self:
        """Compute period-wise difference within each symbol group."""
        result_lf = self._lf.sort(["symbol", "end_time"]).with_columns(
            pl.col("factor").diff(period).over("symbol").alias("factor")
        )

        return self.__class__(result_lf, f"ts_delta({self.name},{period})")

    def ts_rank(self, window: int) -> Self:
        """Rolling rank (percentile rank) with strict NaN semantics (pure Polars)."""
        self._validate_window(window)

        # Create NaN-in-window mask
        nan_in_window = (
            (pl.col("factor").is_null() | pl.col("factor").is_nan())
            .cast(pl.Int64)
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .fill_null(1)
        )

        # Create constant-values mask (std < epsilon)
        rolling_std = pl.col("factor").rolling_std(window_size=window, min_samples=window).over("symbol")

        # Get rolling rank and convert to percentile
        rolling_rank_raw = pl.col("factor").rolling_rank(window_size=window, min_samples=window).over("symbol")

        # Apply masks: NaN if any NaN in window or all values are constant
        rank_expr = (
            pl.when(nan_in_window > 0)
            .then(pl.lit(None))
            .when(rolling_std < EPSILON)
            .then(pl.lit(None))
            .otherwise(rolling_rank_raw / pl.lit(window))
        )

        result_lf = (
            self._lf.sort(["symbol", "end_time"]).with_columns(rank_expr.alias("factor")).sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_rank({self.name},{window})")

    def ts_beta(self, other: Self, window: int) -> Self:
        """
        Time-series Beta from a simple linear regression.
        Beta = Cov(self, other) / Var(other)
        """
        self._validate_window(window)
        self._validate_factor(other, "ts_beta")

        if window < 2:
            result_lf = self._lf.with_columns(pl.lit(None).alias("factor"))
            return self.__class__(result_lf, f"ts_beta({self.name},{other.name},{window})")

        joined = self._lf.join(
            other._lf.rename({"factor": "factor_x"}), on=["start_time", "end_time", "symbol"], how="inner"
        ).sort(["symbol", "end_time"])

        nan_y = (pl.col("factor").is_null() | pl.col("factor").is_nan()).cast(pl.Int64)
        nan_x = (pl.col("factor_x").is_null() | pl.col("factor_x").is_nan()).cast(pl.Int64)
        nan_in_window = (
            (nan_x + nan_y)
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .cast(pl.Int64)
            .fill_null(1)
        )

        mean_x = pl.col("factor_x").rolling_mean(window_size=window, min_samples=window).over("symbol")
        mean_y = pl.col("factor").rolling_mean(window_size=window, min_samples=window).over("symbol")
        mean_xy = (
            (pl.col("factor_x") * pl.col("factor")).rolling_mean(window_size=window, min_samples=window).over("symbol")
        )

        cov_xy = (mean_xy - (mean_x * mean_y)) * pl.lit(window / (window - 1))
        std_x = pl.col("factor_x").rolling_std(window_size=window, min_samples=window, ddof=1).over("symbol")
        var_x = std_x * std_x

        beta_expr = pl.when(var_x.abs() <= EPSILON).then(pl.lit(None)).otherwise(cov_xy / var_x)
        beta_expr = pl.when(nan_in_window > 0).then(pl.lit(None)).otherwise(beta_expr)

        result_lf = joined.with_columns(beta_expr.alias("factor")).drop("factor_x")
        return self.__class__(result_lf, f"ts_beta({self.name},{other.name},{window})")

    def ts_alpha(self, other: Self, window: int) -> Self:
        """
        Time-series Alpha from a simple linear regression.
        Alpha = Mean(self) - Beta * Mean(other)
        """
        self._validate_window(window)
        self._validate_factor(other, "ts_alpha")

        beta = self.ts_beta(other, window)
        alpha = self.ts_mean(window) - beta * other.ts_mean(window)

        return self.__class__(alpha._lf, f"ts_alpha({self.name},{other.name},{window})")

    def ts_resid(self, other: Self, window: int) -> Self:
        """
        Time-series Residual from a simple linear regression.
        Formula: self - (alpha + beta * other)
        """
        self._validate_window(window)
        self._validate_factor(other, "ts_resid")

        alpha = self.ts_alpha(other, window)
        beta = self.ts_beta(other, window)

        result = self - (alpha + beta * other)

        return self.__class__(result._lf, f"ts_resid({self.name},{other.name},{window})")

    def ts_corr(self, other: Self, window: int) -> Self:
        """Rolling correlation using strict NaN semantics (pure Polars)."""
        self._validate_window(window)
        self._validate_factor(other, "ts_corr")

        if window < 2:
            result_lf = self._lf.with_columns(pl.lit(None).alias("factor"))
            return self.__class__(result_lf, f"ts_corr({self.name},{other.name},{window})")

        joined = self._lf.join(
            other._lf.rename({"factor": "factor_y"}), on=["start_time", "end_time", "symbol"], how="inner"
        ).sort(["symbol", "end_time"])

        nan_x = (pl.col("factor").is_null() | pl.col("factor").is_nan()).cast(pl.Int64)
        nan_y = (pl.col("factor_y").is_null() | pl.col("factor_y").is_nan()).cast(pl.Int64)
        nan_in_window = (
            (nan_x + nan_y)
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .cast(pl.Int64)
            .fill_null(1)
        )

        mean_x = pl.col("factor").rolling_mean(window_size=window, min_samples=window).over("symbol")
        mean_y = pl.col("factor_y").rolling_mean(window_size=window, min_samples=window).over("symbol")
        mean_xy = (
            (pl.col("factor") * pl.col("factor_y")).rolling_mean(window_size=window, min_samples=window).over("symbol")
        )

        cov_xy = (mean_xy - (mean_x * mean_y)) * pl.lit(window / (window - 1))
        std_x = pl.col("factor").rolling_std(window_size=window, min_samples=window, ddof=1).over("symbol")
        std_y = pl.col("factor_y").rolling_std(window_size=window, min_samples=window, ddof=1).over("symbol")

        corr_expr = pl.when((std_x <= EPSILON) | (std_y <= EPSILON))
        corr_expr = corr_expr.then(pl.lit(None)).otherwise(cov_xy / (std_x * std_y))
        corr_expr = pl.when(nan_in_window > 0).then(pl.lit(None)).otherwise(corr_expr)

        result_lf = joined.with_columns(corr_expr.alias("factor")).drop("factor_y")
        return self.__class__(result_lf, f"ts_corr({self.name},{other.name},{window})")

    def ts_cov(self, other: Self, window: int) -> Self:
        """Rolling covariance using strict NaN semantics (pure Polars)."""
        self._validate_window(window)
        self._validate_factor(other, "ts_cov")

        if window < 2:
            result_lf = self._lf.with_columns(pl.lit(None).alias("factor"))
            return self.__class__(result_lf, f"ts_cov({self.name},{other.name},{window})")

        joined = self._lf.join(
            other._lf.rename({"factor": "factor_y"}), on=["start_time", "end_time", "symbol"], how="inner"
        ).sort(["symbol", "end_time"])

        nan_x = (pl.col("factor").is_null() | pl.col("factor").is_nan()).cast(pl.Int64)
        nan_y = (pl.col("factor_y").is_null() | pl.col("factor_y").is_nan()).cast(pl.Int64)
        nan_in_window = (
            (nan_x + nan_y)
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .cast(pl.Int64)
            .fill_null(1)
        )

        mean_x = pl.col("factor").rolling_mean(window_size=window, min_samples=window).over("symbol")
        mean_y = pl.col("factor_y").rolling_mean(window_size=window, min_samples=window).over("symbol")
        mean_xy = (
            (pl.col("factor") * pl.col("factor_y")).rolling_mean(window_size=window, min_samples=window).over("symbol")
        )

        cov_expr = (mean_xy - (mean_x * mean_y)) * pl.lit(window / (window - 1))
        cov_expr = pl.when(nan_in_window > 0).then(pl.lit(None)).otherwise(cov_expr)

        result_lf = joined.with_columns(cov_expr.alias("factor")).drop("factor_y")
        return self.__class__(result_lf, f"ts_cov({self.name},{other.name},{window})")

    def ts_cv(self, window: int) -> Self:
        """
        Coefficient of Variation (pure Polars)
        CV = std(x) / |mean(x)|
        """
        self._validate_window(window)

        # Create NaN-in-window mask
        nan_in_window = (
            (pl.col("factor").is_null() | pl.col("factor").is_nan())
            .cast(pl.Int64)
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .fill_null(1)
        )

        # Calculate std and mean
        std_expr = pl.col("factor").rolling_std(window_size=window, min_samples=window, ddof=1).over("symbol")
        mean_expr = pl.col("factor").rolling_mean(window_size=window, min_samples=window).over("symbol")

        # CV = std / |mean| with epsilon to avoid division by zero
        cv_expr = std_expr / (mean_expr.abs() + 1e-10)

        # Apply NaN mask and handle inf
        cv_expr = (
            pl.when(nan_in_window > 0)
            .then(pl.lit(None))
            .when(cv_expr.is_infinite())
            .then(pl.lit(None))
            .otherwise(cv_expr)
        )

        result_lf = (
            self._lf.sort(["symbol", "end_time"]).with_columns(cv_expr.alias("factor")).sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_cv({self.name},{window})")

    def ts_jumpiness(self, window: int) -> Self:
        """
        Compares the total path traveled vs the range (max - min).
        """
        self._validate_window(window)
        diff = self.ts_delta(1).abs()
        total_jump = diff.ts_sum(window)
        range_val = self.ts_max(window) - self.ts_min(window)
        result = total_jump / (range_val + 1e-10)
        result._lf = result._lf.with_columns(
            pl.col("factor").replace(float("inf"), None).replace(float("-inf"), None).alias("factor")
        )
        return self.__class__(result._lf, f"ts_jumpiness({self.name},{window})")

    def ts_autocorr(self, window: int, lag: int = 1) -> Self:
        self._validate_window(window)
        if lag <= 0:
            raise ValueError("Lag must be positive")

        lagged_factor = self.ts_shift(lag)
        result = self.ts_corr(lagged_factor, window)
        return self.__class__(result._lf, f"ts_autocorr({self.name},{window},{lag})")

    def ts_reversal_count(self, window: int) -> Self:
        """Count sign reversals in a rolling window (pure Polars).

        A reversal occurs when consecutive differences change sign.
        Returns the ratio of reversals to possible reversals in the window.
        """
        self._validate_window(window)

        if window < 3:
            # Need at least 3 values to have 2 diffs and 1 possible reversal
            result_lf = self._lf.with_columns(pl.lit(None).alias("factor"))
            return self.__class__(result_lf, f"ts_reversal_count({self.name},{window})")

        # Create NaN-in-window mask (need to check the original factor column)
        nan_in_window = (
            (pl.col("factor").is_null() | pl.col("factor").is_nan())
            .cast(pl.Int64)
            .rolling_max(window_size=window, min_samples=window)
            .over("symbol")
            .fill_null(1)
        )

        # Calculate diff within each symbol group
        diff_expr = pl.col("factor").diff().over("symbol")

        # Sign change indicator: (diff[i] * diff[i-1]) < 0
        # Need to shift the diff within symbol group
        sign_change_expr = ((diff_expr * diff_expr.shift(1).over("symbol")) < 0).cast(pl.Int64)

        # Rolling sum of sign changes
        # For window=N, we have N-1 diffs, and N-2 sign change opportunities
        reversal_sum_expr = sign_change_expr.rolling_sum(window_size=window - 2, min_samples=window - 2).over("symbol")

        # Normalize by window - 2 (number of possible reversals)
        reversal_rate_expr = reversal_sum_expr / pl.lit(window - 2)

        # Apply NaN mask
        result_expr = pl.when(nan_in_window > 0).then(pl.lit(None)).otherwise(reversal_rate_expr)

        result_lf = (
            self._lf.sort(["symbol", "end_time"]).with_columns(result_expr.alias("factor")).sort(["end_time", "symbol"])
        )

        return self.__class__(result_lf, f"ts_reversal_count({self.name},{window})")

    def ts_vr(self, window: int, k: int = 2) -> Self:
        """
        Variance Ratio - tests if the market follows a random walk hypothesis.

        VR ≈ 1: Random walk
        VR > 1: Trending (positive autocorrelation)
        VR < 1: Mean-reverting (negative autocorrelation)
        """
        self._validate_window(window)
        if k <= 0:
            raise ValueError("k must be positive")
        k_diff = self.ts_delta(k)
        one_diff = self.ts_delta(1)
        var_k = k_diff.ts_std(window) ** 2
        var_1 = one_diff.ts_std(window) ** 2
        result = var_k / (k * var_1 + 1e-10)
        result._lf = result._lf.with_columns(
            pl.col("factor").replace(float("inf"), None).replace(float("-inf"), None).alias("factor")
        )
        return self.__class__(result._lf, f"ts_vr({self.name},{window},{k})")
