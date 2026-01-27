import numpy as np
import pandas as pd
import pytest

from factorium import AggBar, Factor
from tests.mixins.test_mathmixin import assert_factor_equals_df


# ==========================================
# Helpers for TimeSeriesOpsMixin
# ==========================================


def emulate_ts_op(factor: Factor, window: int, pandas_func: str) -> pd.Series:
    """
    Generic time-series rolling helper.
    """
    df = factor.to_pandas().copy()

    rolled = (
        df.groupby("symbol")["factor"]
        .rolling(window=window, min_periods=window)
        .agg(pandas_func)
        .reset_index(level=0, drop=True)
    )

    df["factor"] = rolled
    return df["factor"]


def emulate_ts_product(factor: Factor, window: int) -> pd.Series:
    df = factor.to_pandas().copy()

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
    df = factor.to_pandas().copy()
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
    df = factor.to_pandas().copy()
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
    df = factor.to_pandas().copy()
    return df.groupby("symbol").cumcount() + start


def emulate_ts_shift(factor: Factor, period: int) -> pd.Series:
    df = factor.to_pandas().copy()
    return df.groupby("symbol")["factor"].shift(period)


def emulate_ts_delta(factor: Factor, period: int) -> pd.Series:
    df = factor.to_pandas().copy()
    return df.groupby("symbol")["factor"].diff(period)


def emulate_ts_scale(factor: Factor, window: int, constant: float = 0.0) -> pd.Series:
    df = factor.to_pandas().copy()
    grouped = df.groupby("symbol")

    mins = grouped["factor"].transform(lambda s: s.rolling(window=window, min_periods=window).min())
    maxs = grouped["factor"].transform(lambda s: s.rolling(window=window, min_periods=window).max())

    scaled = (df["factor"] - mins) / (maxs - mins)
    scaled = scaled.replace([np.inf, -np.inf], np.nan)
    return scaled + constant


def emulate_ts_zscore(factor: Factor, window: int) -> pd.Series:
    df = factor.to_pandas().copy()
    grouped = df.groupby("symbol")

    means = grouped["factor"].transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    stds = grouped["factor"].transform(lambda s: s.rolling(window=window, min_periods=window).std())

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
    assert res.to_pandas()["factor"].isna().all()


def test_ts_std_constant_series(sample_aggbar: AggBar):
    df = sample_aggbar.to_df()[["start_time", "end_time", "symbol"]].copy()
    df["const"] = 5.0
    fac = Factor(df, name="const")
    window = 3
    res = fac.ts_std(window)
    assert (res.to_pandas()["factor"].fillna(0) == 0).all()


def test_ts_rank_with_nan_in_window(sample_aggbar: AggBar):
    df = sample_aggbar.to_df()[["start_time", "end_time", "symbol", "close"]].copy()
    mask = df["symbol"] == "BTCUSDT"
    first_btc_idx = df[mask].index[0]
    df.loc[first_btc_idx, "close"] = np.nan
    fac = Factor(df, name="close_nan")
    window = 3
    res = fac.ts_rank(window)
    assert res.to_pandas()["factor"].isna().any()


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


@pytest.fixture
def ts_ops_mixin_factory():
    def _factory(series: pd.Series, symbol: str = "BTCUSDT"):
        # Ensure index is datetime for view(np.int64)
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)

        df = pd.DataFrame(
            {
                "start_time": series.index.view(np.int64) // 10**6,
                "end_time": (series.index.view(np.int64) // 10**6) + 60000,
                "symbol": symbol,
                "factor": series.values,
            }
        )
        return Factor(df)

    return _factory


# ==========================================
# Bivariate Tests
# ==========================================


def test_ts_beta(ts_ops_mixin_factory):
    # Setup perfect correlation y = 2x
    # Beta should be 2.0
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2020-01-01", periods=5))
    s2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0], index=pd.date_range("2020-01-01", periods=5))

    f1 = ts_ops_mixin_factory(s1)  # X
    f2 = ts_ops_mixin_factory(s2)  # Y

    # Window=3
    # Var(X) for [1,2,3] is 1.0. Cov(X,Y) is 2.0. Beta = 2.0/1.0 = 2.0
    result = f2.ts_beta(f1, window=3)

    # First 2 should be NaN due to window
    result_series = result.to_pandas()["factor"]
    assert np.isnan(result_series.iloc[0])
    assert np.isnan(result_series.iloc[1])
    assert np.isclose(result_series.iloc[2], 2.0)
    assert np.isclose(result_series.iloc[3], 2.0)
    assert np.isclose(result_series.iloc[4], 2.0)


def test_ts_beta_basic(factor_close: Factor):
    window = 5
    # y = 2 * x
    factor_x = factor_close
    factor_y = factor_close * 2.0

    # ts_beta(y, x, window) = Cov(y, x) / Var(x) = 2.0
    res = factor_y.ts_beta(factor_x, window)

    df = factor_close.to_pandas().copy()
    out = np.full(len(df), np.nan)
    for _, group_idx in df.groupby("symbol").groups.items():
        idx_arr = np.array(list(group_idx))
        if len(idx_arr) >= window:
            out[idx_arr[window - 1 :]] = 2.0

    expected = pd.Series(out)
    assert_factor_equals_df(res, expected)


def test_ts_alpha(ts_ops_mixin_factory):
    # Setup y = x + 5
    # Beta should be 1.0, Alpha should be 5.0
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2020-01-01", periods=5))
    s2 = pd.Series([6.0, 7.0, 8.0, 9.0, 10.0], index=pd.date_range("2020-01-01", periods=5))

    f1 = ts_ops_mixin_factory(s1)  # X
    f2 = ts_ops_mixin_factory(s2)  # Y

    # Window=3
    # Mean(X) for [1,2,3] is 2.0. Mean(Y) for [6,7,8] is 7.0.
    # Beta = 1.0. Alpha = 7.0 - 1.0 * 2.0 = 5.0
    result = f2.ts_alpha(f1, window=3)

    # First 2 should be NaN due to window
    result_series = result.to_pandas()["factor"]
    assert np.isnan(result_series.iloc[0])
    assert np.isnan(result_series.iloc[1])
    assert np.isclose(result_series.iloc[2], 5.0)
    assert np.isclose(result_series.iloc[3], 5.0)
    assert np.isclose(result_series.iloc[4], 5.0)


def test_ts_alpha_nan_handling(ts_ops_mixin_factory):
    # Setup y = x + 5 with a NaN in the middle
    # window = 3
    s1 = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], index=pd.date_range("2020-01-01", periods=5))
    s2 = pd.Series([6.0, 7.0, 8.0, 9.0, 10.0], index=pd.date_range("2020-01-01", periods=5))

    f1 = ts_ops_mixin_factory(s1)  # X
    f2 = ts_ops_mixin_factory(s2)  # Y

    result = f2.ts_alpha(f1, window=3)

    # All should be NaN because:
    # t=0,1: window not full
    # t=2: [1,2,nan]
    # t=3: [2,nan,4]
    # t=4: [nan,4,5]
    assert result.to_pandas()["factor"].isna().all()

    # Case where NaN is at the beginning, so some windows become valid
    s1 = pd.Series([np.nan, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2020-01-01", periods=5))
    s2 = pd.Series([6.0, 7.0, 8.0, 9.0, 10.0], index=pd.date_range("2020-01-01", periods=5))
    f1 = ts_ops_mixin_factory(s1)
    f2 = ts_ops_mixin_factory(s2)
    result = f2.ts_alpha(f1, window=3)

    # t=0,1: window not full
    # t=2: [nan,2,3] -> NaN
    # t=3: [2,3,4] -> OK
    # t=4: [3,4,5] -> OK
    result_series = result.to_pandas()["factor"]
    assert np.isnan(result_series.iloc[0])
    assert np.isnan(result_series.iloc[1])
    assert np.isnan(result_series.iloc[2])
    assert not np.isnan(result_series.iloc[3])
    assert not np.isnan(result_series.iloc[4])


def test_ts_resid(ts_ops_mixin_factory):
    # Setup y = 2x + 3
    # Beta = 2.0, Alpha = 3.0
    # Resid = y - (alpha + beta * x) = (2x + 3) - (3 + 2 * x) = 0
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2020-01-01", periods=5))
    s2 = pd.Series([5.0, 7.0, 9.0, 11.0, 13.0], index=pd.date_range("2020-01-01", periods=5))

    f1 = ts_ops_mixin_factory(s1)  # X
    f2 = ts_ops_mixin_factory(s2)  # Y

    result = f2.ts_resid(f1, window=3)

    # First 2 should be NaN due to window
    result_series = result.to_pandas()["factor"]
    assert np.isnan(result_series.iloc[0])
    assert np.isnan(result_series.iloc[1])
    # Resid should be 0.0
    assert np.allclose(result_series.iloc[2:], 0.0, atol=1e-10)
