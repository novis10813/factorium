import numpy as np
import pandas as pd
import pytest

from factorium import AggBar, Factor


# ==========================================
# Helpers for Dual Verification
# ==========================================


def assert_factor_equals_df(factor_res: Factor, expected_series: pd.Series, check_names=False):
    """
    Verifies that the factor's data matches the expected pandas Series.
    """
    actual_df = factor_res.to_pandas().copy()

    assert len(actual_df) == len(expected_series), "Length mismatch"

    actual_vals = actual_df["factor"].values
    expected_vals = expected_series.values if isinstance(expected_series, pd.Series) else expected_series

    pd.testing.assert_series_equal(
        pd.Series(actual_vals), pd.Series(expected_vals), check_names=False, rtol=1e-5, atol=1e-8
    )


def emulate_unary_op(factor, func):
    """Applies a numpy function to the factor's 'factor' column"""
    df = factor.to_pandas().copy()
    return func(df["factor"])


def emulate_binary_op(f1, f2, func):
    """Applies a binary function between two factors"""
    m = pd.merge(
        f1.to_pandas(),
        f2.to_pandas(),
        on=["start_time", "end_time", "symbol"],
        suffixes=("_x", "_y"),
        how="inner",
    )
    return func(m["factor_x"], m["factor_y"])


def emulate_binary_scalar_op(f1, scalar, func):
    df = f1.to_pandas().copy()
    return func(df["factor"], scalar)


# ==========================================
# Test Cases
# ==========================================


def test_abs(factor_close):
    res = factor_close.abs()
    expected = emulate_unary_op(factor_close, np.abs)
    assert_factor_equals_df(res, expected)


def test_sign(factor_close):
    res = factor_close.sign()
    expected = emulate_unary_op(factor_close, np.sign)
    assert_factor_equals_df(res, expected)


def test_inverse(factor_close):
    res = factor_close.inverse()

    def inv_logic(s):
        return np.where(s != 0, 1 / s, np.nan)

    expected = emulate_unary_op(factor_close, inv_logic)
    assert_factor_equals_df(res, expected)


def test_log(factor_close):
    res = factor_close.log()

    def log_logic(s):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, np.log(s), np.nan)

    expected = emulate_unary_op(factor_close, log_logic)
    assert_factor_equals_df(res, expected)


def test_sqrt(factor_close):
    res = factor_close.sqrt()

    def sqrt_logic(s):
        with np.errstate(invalid="ignore"):
            return np.where(s > 0, np.sqrt(s), np.nan)

    expected = emulate_unary_op(factor_close, sqrt_logic)
    assert_factor_equals_df(res, expected)


def test_signed_log1p(factor_close):
    res = factor_close.signed_log1p()

    def logic(s):
        return np.sign(s) * np.log1p(np.abs(s))


def test_radd_scalar(factor_close):
    """Test 10 + factor"""
    res = 10 + factor_close
    expected = emulate_binary_scalar_op(factor_close, 10, lambda x, y: y + x)
    assert_factor_equals_df(res, expected)


def test_rsub_scalar(factor_close):
    """Test 100 - factor"""
    res = 100 - factor_close
    expected = emulate_binary_scalar_op(factor_close, 100, lambda x, y: y - x)
    assert_factor_equals_df(res, expected)


def test_rmul_scalar(factor_close):
    """Test 2 * factor"""
    res = 2 * factor_close
    expected = emulate_binary_scalar_op(factor_close, 2, lambda x, y: y * x)
    assert_factor_equals_df(res, expected)


def test_rdiv_scalar(factor_close):
    """Test 1000 / factor"""
    res = 1000 / factor_close

    def rdiv_logic(x, y):
        # x is factor series, y is scalar 1000
        # result should be 1000 / x
        return np.where(x != 0, y / x, np.nan)

    expected = emulate_binary_scalar_op(factor_close, 1000, rdiv_logic)
    assert_factor_equals_df(res, expected)


def test_add_scalar(factor_close):
    res = factor_close.add(10)
    expected = emulate_binary_scalar_op(factor_close, 10, lambda x, y: x + y)
    assert_factor_equals_df(res, expected)


def test_add_factor(factor_close, factor_open):
    res = factor_close + factor_open
    expected = emulate_binary_op(factor_close, factor_open, lambda x, y: x + y)
    assert_factor_equals_df(res, expected)


def test_sub_factor(factor_close, factor_open):
    res = factor_close - factor_open
    expected = emulate_binary_op(factor_close, factor_open, lambda x, y: x - y)
    assert_factor_equals_df(res, expected)


def test_mul_factor(factor_close, factor_open):
    res = factor_close * factor_open
    expected = emulate_binary_op(factor_close, factor_open, lambda x, y: x * y)
    assert_factor_equals_df(res, expected)


def test_div_factor(factor_close, factor_open):
    res = factor_close / factor_open

    def safe_div(x, y):
        return np.where(np.abs(y) > 1e-10, x / y, np.nan)

    expected = emulate_binary_op(factor_close, factor_open, safe_div)
    assert_factor_equals_df(res, expected)


def test_max_scalar(factor_close):
    res = factor_close.max(10)
    expected = emulate_binary_scalar_op(factor_close, 10, np.maximum)
    assert_factor_equals_df(res, expected)


def test_min_factor(factor_close, factor_open):
    res = factor_close.min(factor_open)
    expected = emulate_binary_op(factor_close, factor_open, np.minimum)
    assert_factor_equals_df(res, expected)


def test_where(factor_close):
    cond = factor_close > 0
    res = factor_close.where(cond, 999)

    df = factor_close.to_pandas().copy()
    cond_vals = df["factor"] > 0
    expected_vals = np.where(cond_vals, df["factor"], 999)

    assert_factor_equals_df(res, pd.Series(expected_vals))


def test_reverse(factor_close):
    res = factor_close.reverse()
    expected = emulate_unary_op(factor_close, lambda x: -x)
    assert_factor_equals_df(res, expected)


def test_comparison_gt(factor_close):
    res = factor_close > 50
    expected = emulate_binary_scalar_op(factor_close, 50, lambda x, y: (x > y).astype(int))
    assert_factor_equals_df(res, expected)


def test_signed_pow_scalar(factor_close):
    res = factor_close.signed_pow(2)

    def logic(s):
        sign = np.sign(s)
        abs_val = np.abs(s)
        val = sign * (abs_val**2)
        return val.replace([np.inf, -np.inf], np.nan)

    expected = emulate_unary_op(factor_close, logic)
    assert_factor_equals_df(res, expected)
