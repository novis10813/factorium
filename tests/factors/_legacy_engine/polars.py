import numpy as np
import pandas as pd
import polars as pl

from factorium.constants import EPSILON


class PolarsEngine:
    def _to_polars(self, df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
        if isinstance(df, pl.DataFrame):
            return df
        return pl.from_pandas(df)

    def _to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
        return df.to_pandas()

    def _as_pandas(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            return df
        return df.to_pandas()

    def ts_sum(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        result = pl_df.with_columns(
            pl.col(value_col).rolling_sum(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )
        return result

    def ts_mean(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        result = pl_df.with_columns(
            pl.col(value_col).rolling_mean(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )
        return result

    def ts_std(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        result = pl_df.with_columns(
            pl.col(value_col).rolling_std(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )
        return result

    def ts_min(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        result = pl_df.with_columns(
            pl.col(value_col).rolling_min(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )
        return result

    def ts_max(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        result = pl_df.with_columns(
            pl.col(value_col).rolling_max(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )
        return result

    def ts_median(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        result = pl_df.with_columns(
            pl.col(value_col).rolling_median(window_size=window, min_samples=window).over(symbol_col).alias(value_col)
        )
        return result

    def ts_product(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        def safe_prod(x: pd.Series) -> float:
            return np.nan if x.isna().any() else x.prod()

        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .apply(safe_prod, raw=False)
            .reset_index(level=0, drop=True)
        )
        return result

    def ts_rank(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result_df = df.copy()
        out = np.full(len(df), np.nan)

        for _, group_idx in df.groupby(symbol_col).groups.items():
            idx_arr = np.array(list(group_idx))
            vals = df.loc[idx_arr, value_col].to_numpy()

            for i in range(window - 1, len(idx_arr)):
                w = vals[i - window + 1 : i + 1]
                if np.isnan(w).any() or len(np.unique(w)) == 1:
                    continue
                sorted_idx = np.argsort(w)
                rank_array = np.empty_like(sorted_idx, dtype=float)
                rank_array[sorted_idx] = np.arange(1, len(w) + 1)
                out[idx_arr[i]] = rank_array[-1] / len(w)

        result_df[value_col] = out
        return result_df

    def ts_argmin(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result_df = df.copy()
        out = np.full(len(df), np.nan)

        for _, group_idx in df.groupby(symbol_col).groups.items():
            idx_arr = np.array(list(group_idx))
            vals = df.loc[idx_arr, value_col].to_numpy()

            for i in range(window - 1, len(idx_arr)):
                w = vals[i - window + 1 : i + 1]
                if np.isnan(w).any() or len(w) < window:
                    continue
                out[idx_arr[i]] = (len(w) - 1) - np.argmin(w)

        result_df[value_col] = out
        return result_df

    def ts_argmax(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result_df = df.copy()
        out = np.full(len(df), np.nan)

        for _, group_idx in df.groupby(symbol_col).groups.items():
            idx_arr = np.array(list(group_idx))
            vals = df.loc[idx_arr, value_col].to_numpy()

            for i in range(window - 1, len(idx_arr)):
                w = vals[i - window + 1 : i + 1]
                if np.isnan(w).any() or len(w) < window:
                    continue
                out[idx_arr[i]] = (len(w) - 1) - np.argmax(w)

        result_df[value_col] = out
        return result_df

    def ts_shift(
        self,
        df: pd.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        result = pl_df.with_columns(pl.col(value_col).shift(period).over(symbol_col).alias(value_col))
        return result

    def ts_diff(
        self,
        df: pd.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        result = pl_df.with_columns(pl.col(value_col).diff(period).over(symbol_col).alias(value_col))
        return result

    def ts_kurtosis(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result_df = df.copy()
        out = np.full(len(df), np.nan)

        for _, group_idx in df.groupby(symbol_col).groups.items():
            idx_arr = np.array(list(group_idx))
            vals = df.loc[idx_arr, value_col].to_numpy()

            for i in range(window - 1, len(idx_arr)):
                w = vals[i - window + 1 : i + 1]
                if np.isnan(w).any() or len(np.unique(w)) < 2:
                    continue
                mean_val = np.mean(w)
                std_val = np.std(w, ddof=0)
                if std_val < EPSILON:
                    continue
                deviations = w - mean_val
                out[idx_arr[i]] = np.mean(deviations**4) / (std_val**4) - 3

        result_df[value_col] = out
        return result_df

    def ts_skewness(
        self,
        df: pd.DataFrame | pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        def safe_skew(x: pd.Series) -> float:
            if x.isna().any() or len(x) < window:
                return np.nan
            std_val = x.std(ddof=0)
            if std_val < EPSILON:
                return np.nan
            z = (x - x.mean()) / std_val
            return (z**3).mean()

        pdf = self._as_pandas(df)
        result = pdf.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .apply(safe_skew, raw=False)
            .reset_index(level=0, drop=True)
        )
        return pl.from_pandas(result)

    def ts_corr(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        merged = pd.merge(df, other_df, on=["start_time", "end_time", symbol_col], suffixes=("_x", "_y"))
        if merged.empty:
            raise ValueError("No common data between factors")

        result = merged.copy()
        result[value_col] = np.nan

        for symbol, group in result.groupby(symbol_col):
            x = group["factor_x"]
            y = group["factor_y"]

            valid_mask = x.notna() & y.notna()
            if valid_mask.sum() < 2 or x[valid_mask].std() == 0 or y[valid_mask].std() == 0:
                continue

            corr_result = group[["factor_x", "factor_y"]].rolling(window, min_periods=window).corr().iloc[0::2, 1]
            corr_result.index = group.index
            result.loc[group.index, value_col] = corr_result.values

        return result[["start_time", "end_time", symbol_col, value_col]]

    def ts_cov(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        merged = pd.merge(df, other_df, on=["start_time", "end_time", symbol_col], suffixes=("_x", "_y"))
        if merged.empty:
            raise ValueError("No common data between factors")

        result = merged.copy()
        result[value_col] = np.nan

        for symbol, group in result.groupby(symbol_col):
            x = group["factor_x"]
            y = group["factor_y"]

            valid_mask = x.notna() & y.notna()
            if valid_mask.sum() < 2:
                continue

            cov_result = group[["factor_x", "factor_y"]].rolling(window, min_periods=window).cov().iloc[0::2, 1]
            cov_result.index = group.index
            result.loc[group.index, value_col] = cov_result.values

        return result[["start_time", "end_time", symbol_col, value_col]]

    def ts_cv(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result_df = df.copy()
        out = np.full(len(df), np.nan)

        for _, group_idx in df.groupby(symbol_col).groups.items():
            idx_arr = np.array(list(group_idx))
            vals = df.loc[idx_arr, value_col].to_numpy()

            for i in range(window - 1, len(idx_arr)):
                w = vals[i - window + 1 : i + 1]
                if np.isnan(w).any() or len(w) < window:
                    continue
                mean_val = np.mean(w)
                std_val = np.std(w, ddof=1)
                out[idx_arr[i]] = std_val / (abs(mean_val) + EPSILON)

        result_df[value_col] = out
        result_df[value_col] = result_df[value_col].replace([np.inf, -np.inf], np.nan)
        return result_df

    def ts_reversal_count(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        def count_reversals(s: np.ndarray) -> float:
            if len(s) < 3:
                return np.nan
            diff = np.diff(s)
            if len(diff) < 2:
                return np.nan
            valid_diff = diff[~np.isnan(diff)]
            if len(valid_diff) < 2:
                return np.nan
            sign_changes = ((valid_diff[1:] * valid_diff[:-1]) < 0).sum()
            return sign_changes / (len(valid_diff) - 1)

        result_df = df.copy()
        result_df[value_col] = (
            result_df.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=3)
            .apply(count_reversals, raw=True)
            .reset_index(level=0, drop=True)
        )
        return result_df

    def cs_rank(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        rank_expr = (
            pl.col(value_col)
            .rank(method="min", descending=False)
            .over(time_col)
            .truediv(pl.col(value_col).count().over(time_col))
        )
        result = pl_df.with_columns(pl.when(has_nan).then(None).otherwise(rank_expr).alias(value_col))
        return result

    def cs_zscore(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        zscore_expr = (pl.col(value_col) - pl.col(value_col).mean().over(time_col)) / pl.col(value_col).std().over(
            time_col
        )
        result = pl_df.with_columns(pl.when(has_nan).then(None).otherwise(zscore_expr).alias(value_col))
        return result

    def cs_demean(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        demean_expr = pl.col(value_col) - pl.col(value_col).mean().over(time_col)
        result = pl_df.with_columns(pl.when(has_nan).then(None).otherwise(demean_expr).alias(value_col))
        return result

    def to_pandas(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        return self._to_pandas(pl_df)

    def from_pandas(self, df: pd.DataFrame) -> pl.DataFrame:
        return pl.from_pandas(df)

    def cs_winsorize(
        self,
        df: pd.DataFrame,
        lower_limit: float,
        upper_limit: float,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result_df = df.copy()
        out = np.full(len(df), np.nan)

        for time_val, group in df.groupby(time_col):
            vals = group[value_col]
            if vals.isna().any():
                continue
            lower_val = vals.quantile(lower_limit)
            upper_val = vals.quantile(1 - upper_limit)
            clipped = vals.clip(lower=lower_val, upper=upper_val)
            out[group.index] = clipped.values

        result_df[value_col] = out
        return result_df

    def cs_mean(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Cross-sectional mean. Strict: Returns NaN if any input is NaN."""
        pl_df = self._to_polars(df)
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        mean_expr = pl.col(value_col).mean().over(time_col)
        result = pl_df.with_columns(pl.when(has_nan).then(None).otherwise(mean_expr).alias(value_col))
        return self._to_pandas(result)

    def cs_median(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Cross-sectional median. Strict: Returns NaN if any input is NaN."""
        pl_df = self._to_polars(df)
        has_nan = pl.col(value_col).is_null().any().over(time_col)
        median_expr = pl.col(value_col).median().over(time_col)
        result = pl_df.with_columns(pl.when(has_nan).then(None).otherwise(median_expr).alias(value_col))
        return self._to_pandas(result)

    def cs_neutralize(
        self,
        df_y: pd.DataFrame,
        df_x: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Cross-sectional neutralization via least-squares regression.

        Returns residuals of: Y = alpha + beta * X + residuals
        Strict: Returns NaN if any input is NaN in the cross-section.
        """
        # Merge Y and X data on time and symbol
        merged = pd.merge(df_y, df_x, on=["start_time", time_col, "symbol"], suffixes=("_y", "_x"))

        if merged.empty:
            raise ValueError("No common data for neutralization")

        # Group by time period and compute residuals
        residual_series_list = []

        for _, group in merged.groupby(time_col):
            # Check for NaNs strictly in the whole group (both x and y)
            if group[[f"{value_col}_y", f"{value_col}_x"]].isna().any().any():
                residual_series_list.append(pd.Series(np.nan, index=group.index))
                continue

            y = group[f"{value_col}_y"].values.astype(float)
            x = group[f"{value_col}_x"].values.astype(float)

            # Check for constant x (cannot regress)
            if np.std(x) < EPSILON:
                residual_series_list.append(pd.Series(np.nan, index=group.index))
                continue

            try:
                # Solve least squares: y = [x 1] * [beta alpha]^T
                A = np.vstack([x, np.ones(len(x))]).T
                beta_alpha, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                residuals = y - A @ beta_alpha
                residual_series_list.append(pd.Series(residuals, index=group.index))
            except Exception:
                residual_series_list.append(pd.Series(np.nan, index=group.index))

        result_data = merged.copy()
        if residual_series_list:
            full_resid = pd.concat(residual_series_list)
            # Align by index
            result_data[value_col] = full_resid
        else:
            result_data[value_col] = np.nan

        result_data = result_data[["start_time", time_col, "symbol", value_col]]
        return result_data

    def abs(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        result = df.copy()
        result[value_col] = np.abs(result[value_col])
        return result

    def sign(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        result = df.copy()
        result[value_col] = np.sign(result[value_col])
        return result

    def log(self, df: pd.DataFrame, base: float | None = None, value_col: str = "factor") -> pd.DataFrame:
        result = df.copy()
        vals = result[value_col]
        mask = vals > 0

        if base is None:
            log_vals = np.log(vals[mask])
        else:
            if base <= 0 or base == 1:
                raise ValueError(f"Invalid log base: {base}")
            log_vals = np.log(vals[mask]) / np.log(base)

        result[value_col] = np.nan
        result.loc[mask, value_col] = log_vals
        return result

    def sqrt(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        result = df.copy()
        with np.errstate(invalid="ignore"):
            result[value_col] = np.where(result[value_col] > 0, np.sqrt(result[value_col]), np.nan)
        return result

    def pow(self, df: pd.DataFrame, exponent: float, value_col: str = "factor") -> pd.DataFrame:
        result = df.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            result[value_col] = result[value_col] ** exponent
        result[value_col] = result[value_col].replace([np.inf, -np.inf], np.nan)
        return result

    def neg(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        result = df.copy()
        result[value_col] = -result[value_col]
        return result

    def inverse(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        result = df.copy()
        result[value_col] = np.where(result[value_col] != 0, 1 / result[value_col], np.nan)
        return result

    def signed_pow(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        pl_other = self._to_polars(other_df)

        merged = pl_df.join(
            pl_other.select(["start_time", "end_time", "symbol", pl.col(value_col).alias("other")]),
            on=["start_time", "end_time", "symbol"],
            how="inner",
        )

        result = merged.with_columns(
            (pl.col(value_col).sign() * pl.col(value_col).abs().pow(pl.col("other"))).alias(value_col)
        ).drop("other")

        return self._to_pandas(result)

    def maximum(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        pl_other = self._to_polars(other_df)

        merged = pl_df.join(
            pl_other.select(["start_time", "end_time", "symbol", pl.col(value_col).alias("other")]),
            on=["start_time", "end_time", "symbol"],
            how="inner",
        )

        result = merged.with_columns(pl.max_horizontal(value_col, "other").alias(value_col)).drop("other")

        return self._to_pandas(result)

    def minimum(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        pl_other = self._to_polars(other_df)

        merged = pl_df.join(
            pl_other.select(["start_time", "end_time", "symbol", pl.col(value_col).alias("other")]),
            on=["start_time", "end_time", "symbol"],
            how="inner",
        )

        result = merged.with_columns(pl.min_horizontal(value_col, "other").alias(value_col)).drop("other")

        return self._to_pandas(result)

    def where(
        self,
        df: pd.DataFrame,
        cond_df: pd.DataFrame,
        other_df: pd.DataFrame | None,
        other_scalar: float | None = None,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        pl_df = self._to_polars(df)
        pl_cond = self._to_polars(cond_df)

        merged = pl_df.join(
            pl_cond.select(["start_time", "end_time", "symbol", pl.col(value_col).alias("cond")]),
            on=["start_time", "end_time", "symbol"],
            how="inner",
        )

        if other_df is not None:
            pl_other = self._to_polars(other_df)
            merged = merged.join(
                pl_other.select(["start_time", "end_time", "symbol", pl.col(value_col).alias("other")]),
                on=["start_time", "end_time", "symbol"],
                how="inner",
            )
            result = merged.with_columns(
                pl.when(pl.col("cond").fill_null(False).cast(pl.Boolean))
                .then(pl.col(value_col))
                .otherwise(pl.col("other"))
                .alias(value_col)
            ).drop(["cond", "other"])
        else:
            other_val = other_scalar if other_scalar is not None else float("nan")
            result = merged.with_columns(
                pl.when(pl.col("cond").fill_null(False).cast(pl.Boolean))
                .then(pl.col(value_col))
                .otherwise(pl.lit(other_val))
                .alias(value_col)
            ).drop("cond")

        return self._to_pandas(result)
