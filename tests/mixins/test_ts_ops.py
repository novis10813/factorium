import pytest
import pandas as pd
import numpy as np

from factorium import Factor, AggBar

from tests.mixins.test_mathmixin import (
    sample_aggbar,  # noqa: F401
    factor_close,  # noqa: F401
    factor_open,  # noqa: F401
    assert_factor_equals_df,
)


# ==========================================
# Helpers for TimeSeriesOpsMixin
# ==========================================


def emulate_ts_op(factor: Factor, window: int, pandas_func: str) -> pd.Series:
    """
    Generic time-series rolling helper.
    """
    df = factor.data.copy()

    rolled = (
        df.groupby("symbol")["factor"]
        .rolling(window=window, min_periods=window)
        .agg(pandas_func)
        .reset_index(level=0, drop=True)
    )

    df["factor"] = rolled
    return df["factor"]


def emulate_ts_product(factor: Factor, window: int) -> pd.Series:
    df = factor.data.copy()

    def safe_prod(s: pd.Series) -> float:
        return np.nan if s.isna().any() else s.prod()

    rolled = (
        df.groupby("symbol")["factor"]
        .rolling(window=window, min_periods=window)
        .apply(safe_prod, raw=False)
        .reset_index(level=0, drop=True)
    )
    df["factor"] = rolled
    return df["factor"]


def emulate_ts_rank(factor: Factor, window: int) -> pd.Series:
    """
    Emulates TimeSeriesOpsMixin.ts_rank logic.
    """
    df = factor.data.copy()
    out = np.full(len(df), np.nan)

    for _, group_idx in df.groupby("symbol").groups.items():
        idx_arr = np.array(list(group_idx))
        vals = df.loc[idx_arr, "factor"].to_numpy()

        for i in range(window - 1, len(idx_arr)):
            w = vals[i - window + 1 : i + 1]
            if np.isnan(w).any() or len(np.unique(w)) == 1:
                continue
            sorted_idx = np.argsort(w)
            rank_array = np.empty_like(sorted_idx, dtype=float)
            rank_array[sorted_idx] = np.arange(1, len(w) + 1)
            out[idx_arr[i]] = rank_array[-1] / len(w)

    return pd.Series(out)


def emulate_ts_argminmax(factor: Factor, window: int, is_min: bool) -> pd.Series:
    """
    Emulates ts_argmin / ts_argmax.
    """
    df = factor.data.copy()
    out = np.full(len(df), np.nan)

    for _, group_idx in df.groupby("symbol").groups.items():
        idx_arr = np.array(list(group_idx))
        vals = df.loc[idx_arr, "factor"].to_numpy()

        for i in range(window - 1, len(idx_arr)):
            w = vals[i - window + 1 : i + 1]
            if np.isnan(w).any() or len(w) < window:
                continue
            pos = np.argmin(w) if is_min else np.argmax(w)
            out[idx_arr[i]] = (len(w) - 1) - pos

    return pd.Series(out)


def emulate_ts_step(factor: Factor, start: int = 1) -> pd.Series:
    df = factor.data.copy()
    return df.groupby("symbol").cumcount() + start


def emulate_ts_shift(factor: Factor, period: int) -> pd.Series:
    df = factor.data.copy()
    return df.groupby("symbol")["factor"].shift(period)


def emulate_ts_delta(factor: Factor, period: int) -> pd.Series:
    df = factor.data.copy()
    return df.groupby("symbol")["factor"].diff(period)


def emulate_ts_scale(factor: Factor, window: int, constant: float = 0.0) -> pd.Series:
    df = factor.data.copy()
    grouped = df.groupby("symbol")

    mins = grouped["factor"].transform(
        lambda s: s.rolling(window=window, min_periods=window).min()
    )
    maxs = grouped["factor"].transform(
        lambda s: s.rolling(window=window, min_periods=window).max()
    )

    scaled = (df["factor"] - mins) / (maxs - mins)
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    return scaled + constant


def emulate_ts_zscore(factor: Factor, window: int) -> pd.Series:
    df = factor.data.copy()
    grouped = df.groupby("symbol")

    means = grouped["factor"].transform(
        lambda s: s.rolling(window=window, min_periods=window).mean()
    )
    stds = grouped["factor"].transform(
        lambda s: s.rolling(window=window, min_periods=window).std()
    )

    z = (df["factor"] - means) / stds
    z = z.replace([np.inf, -np.inf], np.nan)
    return z


# ==========================================
# Basic Statistics Tests
# ==========================================


def test_ts_sum_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_sum(window)
    expected = emulate_ts_op(factor_close, window, "sum")
    assert_factor_equals_df(res, expected)


def test_ts_mean_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_mean(window)
    expected = emulate_ts_op(factor_close, window, "mean")
    assert_factor_equals_df(res, expected)


def test_ts_median_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_median(window)
    expected = emulate_ts_op(factor_close, window, "median")
    assert_factor_equals_df(res, expected)


def test_ts_std_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_std(window)
    expected = emulate_ts_op(factor_close, window, "std")
    assert_factor_equals_df(res, expected)


def test_ts_min_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_min(window)
    expected = emulate_ts_op(factor_close, window, "min")
    assert_factor_equals_df(res, expected)


def test_ts_max_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_max(window)
    expected = emulate_ts_op(factor_close, window, "max")
    assert_factor_equals_df(res, expected)


def test_ts_product_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_product(window)
    expected = emulate_ts_product(factor_close, window)
    assert_factor_equals_df(res, expected)


# ==========================================
# Edge Cases
# ==========================================


def test_ts_mean_window_larger_than_length(sample_aggbar: AggBar):
    df = sample_aggbar.to_df()[["start_time", "end_time", "symbol", "close"]]
    fac = Factor(df, name="x")
    window = len(df) + 5
    res = fac.ts_mean(window)
    assert res.data["factor"].isna().all()


def test_ts_std_constant_series(sample_aggbar: AggBar):
    df = sample_aggbar.to_df()[["start_time", "end_time", "symbol"]].copy()
    df["const"] = 5.0
    fac = Factor(df, name="const")
    window = 3
    res = fac.ts_std(window)
    assert (res.data["factor"].fillna(0) == 0).all()


def test_ts_rank_with_nan_in_window(sample_aggbar: AggBar):
    df = sample_aggbar.to_df()[["start_time", "end_time", "symbol", "close"]].copy()
    mask = df["symbol"] == "BTCUSDT"
    first_btc_idx = df[mask].index[0]
    df.loc[first_btc_idx, "close"] = np.nan
    fac = Factor(df, name="close_nan")
    window = 3
    res = fac.ts_rank(window)
    assert res.data["factor"].isna().any()


# ==========================================
# Error Cases
# ==========================================


def test_ts_ops_invalid_window_raises(sample_aggbar: AggBar):
    df = sample_aggbar.to_df()[["start_time", "end_time", "symbol", "close"]]
    fac = Factor(df, name="x")
    with pytest.raises(ValueError):
        fac.ts_mean(0)


def test_ts_quantile_invalid_driver_raises(factor_close: Factor):
    with pytest.raises(ValueError):
        factor_close.ts_quantile(3, driver="invalid_driver")


def test_ts_autocorr_invalid_lag_raises(factor_close: Factor):
    with pytest.raises(ValueError):
        factor_close.ts_autocorr(3, lag=0)


def test_ts_vr_invalid_k_raises(factor_close: Factor):
    with pytest.raises(ValueError):
        factor_close.ts_vr(3, k=0)


# ==========================================
# Ranking & Position Tests
# ==========================================


def test_ts_rank_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_rank(window)
    expected = emulate_ts_rank(factor_close, window)
    assert_factor_equals_df(res, expected)


def test_ts_argmin_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_argmin(window)
    expected = emulate_ts_argminmax(factor_close, window, is_min=True)
    assert_factor_equals_df(res, expected)


def test_ts_argmax_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_argmax(window)
    expected = emulate_ts_argminmax(factor_close, window, is_min=False)
    assert_factor_equals_df(res, expected)


# ==========================================
# Normalization Tests
# ==========================================


def test_ts_scale_basic(factor_close: Factor):
    window = 3
    const = 0.5
    res = factor_close.ts_scale(window, constant=const)
    expected = emulate_ts_scale(factor_close, window, constant=const)
    assert_factor_equals_df(res, expected)


def test_ts_zscore_basic(factor_close: Factor):
    window = 3
    res = factor_close.ts_zscore(window)
    expected = emulate_ts_zscore(factor_close, window)
    assert_factor_equals_df(res, expected)


@pytest.mark.parametrize("driver", ["gaussian", "uniform", "cauchy"])
def test_ts_quantile_basic(factor_close: Factor, driver: str):
    window = 3
    res = factor_close.ts_quantile(window, driver=driver)

    from scipy.stats import norm, uniform, cauchy

    ppf_map = {
        "gaussian": norm.ppf,
        "uniform": uniform.ppf,
        "cauchy": cauchy.ppf,
    }

    ranked = emulate_ts_rank(factor_close, window)
    epsilon = 1e-6
    clipped = ranked.clip(lower=epsilon, upper=1 - epsilon)
    expected = ppf_map[driver](clipped)

    assert_factor_equals_df(res, expected)
