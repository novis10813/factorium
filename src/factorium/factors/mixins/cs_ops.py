from typing import List, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numbers
import polars as pl
from ...constants import EPSILON


class CrossSectionalOpsMixin:
    def _strict_nan_mask(self) -> pl.Expr:
        return (pl.col("factor").is_null() | pl.col("factor").is_nan()).any().over("end_time")

    def cs_rank(self) -> Self:
        """Cross-sectional rank (percentile). Strict: Returns NaN if any input is NaN."""
        nan_mask = self._strict_nan_mask()
        rank_expr = pl.col("factor").rank(method="min").over("end_time")
        count_expr = pl.col("factor").count().over("end_time")
        result_lf = self._lf.with_columns(
            pl.when(nan_mask).then(pl.lit(None)).otherwise(rank_expr / count_expr).alias("factor")
        )
        return self.__class__(result_lf, "cs_rank")

    def rank(self) -> Self:
        """Alias for cs_rank."""
        return self.cs_rank()

    def cs_zscore(self) -> Self:
        """Cross-sectional z-score standardization. Strict: Returns NaN if any input is NaN."""
        nan_mask = self._strict_nan_mask()
        mean_expr = pl.col("factor").mean().over("end_time")
        std_expr = pl.col("factor").std(ddof=1).over("end_time")
        z_expr = (pl.col("factor") - mean_expr) / std_expr
        result_lf = self._lf.with_columns(pl.when(nan_mask).then(pl.lit(None)).otherwise(z_expr).alias("factor"))
        return self.__class__(result_lf, "cs_zscore")

    def cs_demean(self) -> Self:
        """Cross-sectional de-meaning. Strict: Returns NaN if any input is NaN."""
        nan_mask = self._strict_nan_mask()
        mean_expr = pl.col("factor").mean().over("end_time")
        result_lf = self._lf.with_columns(
            pl.when(nan_mask).then(pl.lit(None)).otherwise(pl.col("factor") - mean_expr).alias("factor")
        )
        return self.__class__(result_lf, "cs_demean")

    def cs_winsorize(self, limits: Union[float, List[float]] = 0.025) -> Self:
        """
        Cross-sectional winsorization. Strict: Returns NaN if any input is NaN.
        Limits can be a single float (applied to both sides) or [lower, upper].
        """
        if isinstance(limits, numbers.Real):
            lower_lim = upper_lim = limits
        else:
            lower_lim, upper_lim = limits

        nan_mask = self._strict_nan_mask()
        lower_expr = pl.col("factor").quantile(lower_lim).over("end_time")
        upper_expr = pl.col("factor").quantile(1 - upper_lim).over("end_time")
        clipped = pl.col("factor").clip(lower_expr, upper_expr)

        result_lf = self._lf.with_columns(pl.when(nan_mask).then(pl.lit(None)).otherwise(clipped).alias("factor"))
        return self.__class__(result_lf, f"cs_winsorize({limits})")

    def cs_neutralize(self, other: Self) -> Self:
        """
        Cross-sectional neutralization against another factor.
        Returns the residuals of: self = alpha + beta * other + residuals.
        Strict: Returns NaN if any input (self or other) is NaN in the cross-section.

        Pure Polars implementation using:
        beta = Cov(x, y) / Var(x)
        residual = (y - mean_y) - beta * (x - mean_x)
        """
        self._validate_factor(other, "cs_neutralize")

        left = self._lf
        right = other._lf.rename({"factor": "factor_x"})
        joined = left.join(right, on=["start_time", "end_time", "symbol"], how="inner")

        # Check for NaN in either factor within each cross-section
        nan_mask = (pl.col("factor").is_null() | pl.col("factor").is_nan()).any().over("end_time") | (
            pl.col("factor_x").is_null() | pl.col("factor_x").is_nan()
        ).any().over("end_time")

        # Calculate OLS components per cross-section
        mean_y = pl.col("factor").mean().over("end_time")
        mean_x = pl.col("factor_x").mean().over("end_time")
        var_x = pl.col("factor_x").var(ddof=0).over("end_time")
        cov_xy = (pl.col("factor") * pl.col("factor_x")).mean().over("end_time") - mean_x * mean_y

        # beta = Cov(x, y) / Var(x)
        beta = pl.when(var_x.abs() <= EPSILON).then(pl.lit(None)).otherwise(cov_xy / var_x)

        # residual = (y - mean_y) - beta * (x - mean_x)
        residual = (pl.col("factor") - mean_y) - beta * (pl.col("factor_x") - mean_x)

        # Apply NaN mask
        result_expr = pl.when(nan_mask).then(pl.lit(None)).otherwise(residual)

        result_lf = joined.with_columns(result_expr.alias("factor")).select(
            ["start_time", "end_time", "symbol", "factor"]
        )
        return self.__class__(result_lf, f"cs_neutralize({self.name},{other.name})")

    def mean(self) -> Self:
        """Cross-sectional mean. Strict: Returns NaN if any input is NaN."""
        nan_mask = self._strict_nan_mask()
        mean_expr = pl.col("factor").mean().over("end_time")
        result_lf = self._lf.with_columns(pl.when(nan_mask).then(pl.lit(None)).otherwise(mean_expr).alias("factor"))
        return self.__class__(result_lf, "mean")

    def median(self) -> Self:
        """Cross-sectional median. Strict: Returns NaN if any input is NaN."""
        nan_mask = self._strict_nan_mask()
        median_expr = pl.col("factor").median().over("end_time")
        result_lf = self._lf.with_columns(pl.when(nan_mask).then(pl.lit(None)).otherwise(median_expr).alias("factor"))
        return self.__class__(result_lf, "median")
