import numpy as np
import pandas as pd

from factorium.constants import EPSILON


class PandasEngine:
    def ts_sum(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .sum()
            .reset_index(level=0, drop=True)
        )
        return result

    def ts_mean(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(level=0, drop=True)
        )
        return result

    def ts_std(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )
        return result

    def ts_min(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .min()
            .reset_index(level=0, drop=True)
        )
        return result

    def ts_max(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .max()
            .reset_index(level=0, drop=True)
        )
        return result

    def ts_median(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .median()
            .reset_index(level=0, drop=True)
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
        result = df.copy()
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

        result[value_col] = out
        return result

    def ts_argmin(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        out = np.full(len(df), np.nan)

        for _, group_idx in df.groupby(symbol_col).groups.items():
            idx_arr = np.array(list(group_idx))
            vals = df.loc[idx_arr, value_col].to_numpy()

            for i in range(window - 1, len(idx_arr)):
                w = vals[i - window + 1 : i + 1]
                if np.isnan(w).any() or len(w) < window:
                    continue
                out[idx_arr[i]] = (len(w) - 1) - np.argmin(w)

        result[value_col] = out
        return result

    def ts_argmax(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        out = np.full(len(df), np.nan)

        for _, group_idx in df.groupby(symbol_col).groups.items():
            idx_arr = np.array(list(group_idx))
            vals = df.loc[idx_arr, value_col].to_numpy()

            for i in range(window - 1, len(idx_arr)):
                w = vals[i - window + 1 : i + 1]
                if np.isnan(w).any() or len(w) < window:
                    continue
                out[idx_arr[i]] = (len(w) - 1) - np.argmax(w)

        result[value_col] = out
        return result

    def ts_shift(
        self,
        df: pd.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        result[value_col] = result.groupby(symbol_col)[value_col].shift(period)
        return result

    def ts_diff(
        self,
        df: pd.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        result[value_col] = result.groupby(symbol_col)[value_col].diff(period)
        return result

    def ts_kurtosis(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
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

        result[value_col] = out
        return result

    def ts_skewness(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        def safe_skew(x: pd.Series) -> float:
            if x.isna().any() or len(x) < window:
                return np.nan
            std_val = x.std(ddof=0)
            if std_val < EPSILON:
                return np.nan
            z = (x - x.mean()) / std_val
            return (z**3).mean()

        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=window)
            .apply(safe_skew, raw=False)
            .reset_index(level=0, drop=True)
        )
        return result

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
        result = df.copy()
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

        result[value_col] = out
        result[value_col] = result[value_col].replace([np.inf, -np.inf], np.nan)
        return result

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

        result = df.copy()
        result[value_col] = (
            result.groupby(symbol_col)[value_col]
            .rolling(window, min_periods=3)
            .apply(count_reversals, raw=True)
            .reset_index(level=0, drop=True)
        )
        return result

    def cs_rank(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()

        def rank_group(group: pd.DataFrame) -> pd.Series:
            vals = group[value_col]
            if vals.isna().any():
                return pd.Series(np.nan, index=group.index)
            return vals.rank(method="min") / len(vals)

        result[value_col] = (
            result.groupby(time_col, group_keys=False)
            .apply(rank_group, include_groups=False)
            .reset_index(level=0, drop=True)
        )
        return result

    def cs_zscore(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()

        def zscore_group(group: pd.DataFrame) -> pd.Series:
            vals = group[value_col]
            if vals.isna().any():
                return pd.Series(np.nan, index=group.index)
            return (vals - vals.mean()) / vals.std()

        result[value_col] = (
            result.groupby(time_col, group_keys=False)
            .apply(zscore_group, include_groups=False)
            .reset_index(level=0, drop=True)
        )
        return result

    def cs_demean(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()

        def demean_group(group: pd.DataFrame) -> pd.Series:
            vals = group[value_col]
            if vals.isna().any():
                return pd.Series(np.nan, index=group.index)
            return vals - vals.mean()

        result[value_col] = (
            result.groupby(time_col, group_keys=False)
            .apply(demean_group, include_groups=False)
            .reset_index(level=0, drop=True)
        )
        return result

    def cs_winsorize(
        self,
        df: pd.DataFrame,
        lower_limit: float,
        upper_limit: float,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        result = df.copy()
        out = np.full(len(df), np.nan)

        for time_val, group in df.groupby(time_col):
            vals = group[value_col]
            if vals.isna().any():
                continue
            lower_val = vals.quantile(lower_limit)
            upper_val = vals.quantile(1 - upper_limit)
            clipped = vals.clip(lower=lower_val, upper=upper_val)
            out[group.index] = clipped.values

        result[value_col] = out
        return result

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
        merged = pd.merge(df, other_df, on=["start_time", "end_time", "symbol"], suffixes=("", "_other"))

        sign = np.sign(merged[value_col])
        abs_val = np.abs(merged[value_col])

        with np.errstate(divide="ignore", invalid="ignore"):
            result_val = sign * (abs_val ** merged[f"{value_col}_other"])

        merged[value_col] = result_val.replace([np.inf, -np.inf], np.nan)
        return merged[["start_time", "end_time", "symbol", value_col]]

    def maximum(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        merged = pd.merge(df, other_df, on=["start_time", "end_time", "symbol"], suffixes=("", "_other"))
        merged[value_col] = np.maximum(merged[value_col], merged[f"{value_col}_other"])
        return merged[["start_time", "end_time", "symbol", value_col]]

    def minimum(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        merged = pd.merge(df, other_df, on=["start_time", "end_time", "symbol"], suffixes=("", "_other"))
        merged[value_col] = np.minimum(merged[value_col], merged[f"{value_col}_other"])
        return merged[["start_time", "end_time", "symbol", value_col]]

    def where(
        self,
        df: pd.DataFrame,
        cond_df: pd.DataFrame,
        other_df: pd.DataFrame | None,
        other_scalar: float | None = None,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        merged = pd.merge(df, cond_df, on=["start_time", "end_time", "symbol"], suffixes=("", "_cond"))
        cond_bool = merged[f"{value_col}_cond"].fillna(False).astype(bool)

        if other_df is not None:
            merged = pd.merge(
                merged,
                other_df.rename(columns={value_col: f"{value_col}_other"}),
                on=["start_time", "end_time", "symbol"],
            )
            merged[value_col] = np.where(cond_bool, merged[value_col], merged[f"{value_col}_other"])
        else:
            other_val = other_scalar if other_scalar is not None else np.nan
            merged[value_col] = np.where(cond_bool, merged[value_col], other_val)

        return merged[["start_time", "end_time", "symbol", value_col]]
