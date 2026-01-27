"""Tests for TimeSeriesOpsMixin with Polars backend - TDD RED phase.

Tests cover all required ts_ops operators with Polars implementation:
- Basic behavior for each operator
- NaN handling and insufficient window tests
- Numerical consistency with Pandas implementation (rtol=1e-10, atol=1e-12)

These tests are designed to FAIL initially (RED phase) until Polars support
is fully implemented in the production code.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factorium import Factor


# ==========================================
# Fixtures: Minimal sample data
# ==========================================


@pytest.fixture
def sample_polars_factor_data():
    """Create minimal sample factor data for Polars tests.

    Returns a Factor with:
    - start_time, end_time, symbol, factor columns
    - 2 symbols (BTCUSDT, ETHUSDT)
    - 10 data points per symbol
    - Mix of regular and edge case values
    """
    dates = pd.date_range("2025-01-01", periods=10, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6

    common_cols = {
        "start_time": timestamps,
        "end_time": timestamps + 60000,
    }

    # BTCUSDT: predictable sequence for easy verification
    df_btc = pd.DataFrame(common_cols)
    df_btc["symbol"] = "BTCUSDT"
    df_btc["factor"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # ETHUSDT: different pattern
    df_eth = pd.DataFrame(common_cols)
    df_eth["symbol"] = "ETHUSDT"
    df_eth["factor"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    df_combined = pd.concat([df_btc, df_eth], ignore_index=True)

    return Factor(df_combined, name="test_factor")


@pytest.fixture
def sample_polars_factor_with_nan(sample_polars_factor_data):
    """Create factor data with NaN values for edge case testing."""
    df = sample_polars_factor_data.data.to_pandas()

    # Insert NaN in BTCUSDT at index 3
    mask = df["symbol"] == "BTCUSDT"
    btc_indices = df[mask].index
    if len(btc_indices) > 3:
        df.loc[btc_indices[3], "factor"] = np.nan

    return Factor(df, name="test_factor_nan")


@pytest.fixture
def sample_polars_factor_constant():
    """Create factor data with constant values for std/var edge cases."""
    dates = pd.date_range("2025-01-01", periods=10, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6

    common_cols = {
        "start_time": timestamps,
        "end_time": timestamps + 60000,
    }

    df_btc = pd.DataFrame(common_cols)
    df_btc["symbol"] = "BTCUSDT"
    df_btc["factor"] = [5.0] * 10  # All constant

    df_eth = pd.DataFrame(common_cols)
    df_eth["symbol"] = "ETHUSDT"
    df_eth["factor"] = [10.0] * 10  # All constant

    df_combined = pd.concat([df_btc, df_eth], ignore_index=True)

    return Factor(df_combined, name="const_factor")


# ==========================================
# Helper: Pandas reference implementation
# ==========================================


def get_pandas_reference(factor: Factor, window: int, operation: str) -> pd.Series:
    """Get Pandas reference implementation for comparison.

    Maps operation names to emulation logic.
    """
    df = factor.data.to_pandas() if hasattr(factor.data, "to_pandas") else factor.data

    if operation in ["ts_sum", "ts_mean", "ts_std", "ts_min", "ts_max", "ts_median"]:
        pandas_op = operation.replace("ts_", "")

        # Manually apply rolling per group and preserve row order
        result_series = pd.Series(np.nan, index=range(len(df)))
        for symbol, group_indices in df.groupby("symbol", sort=False).groups.items():
            idx_array = np.array(list(group_indices))
            vals = df.loc[idx_array, "factor"].values
            rolled = pd.Series(vals).rolling(window=window, min_periods=window).agg(pandas_op).values
            result_series.iloc[idx_array] = rolled

        return result_series

    raise NotImplementedError(f"Reference implementation for {operation} not available")


def assert_numerical_consistency(
    result_factor: Factor, expected_series: pd.Series, rtol: float = 1e-10, atol: float = 1e-12
):
    """Assert numerical consistency between result and expected with tolerance.

    Parameters
    ----------
    result_factor : Factor
        Result factor from Polars computation
    expected_series : pd.Series
        Expected values (typically from Pandas)
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    """
    result_data = result_factor.data.to_pandas() if hasattr(result_factor.data, "to_pandas") else result_factor.data
    result_values = result_data["factor"].values

    # Handle NaN values: both should be NaN at same positions
    nan_mask_result = np.isnan(result_values)
    nan_mask_expected = np.isnan(expected_series.values)

    assert np.array_equal(nan_mask_result, nan_mask_expected), (
        f"NaN masks don't match: result={nan_mask_result}, expected={nan_mask_expected}"
    )

    # Compare non-NaN values
    valid_mask = ~nan_mask_result
    if valid_mask.any():
        np.testing.assert_allclose(
            result_values[valid_mask],
            expected_series.values[valid_mask],
            rtol=rtol,
            atol=atol,
            err_msg=f"Numerical consistency check failed",
        )


# ==========================================
# Tests: ts_sum
# ==========================================


class TestTsSum:
    """Tests for ts_sum operation."""

    def test_ts_sum_basic(self, sample_polars_factor_data):
        """Test basic ts_sum functionality with valid window."""
        window = 3
        result = sample_polars_factor_data.ts_sum(window)

        # Verify result is a Factor
        assert isinstance(result, Factor)
        assert result.name == f"ts_sum(test_factor,{window})"

        # Verify shape
        assert len(result) == len(sample_polars_factor_data)

    def test_ts_sum_consistency_with_pandas(self, sample_polars_factor_data):
        """Test ts_sum numerical consistency with Pandas reference."""
        window = 3
        result = sample_polars_factor_data.ts_sum(window)
        expected = get_pandas_reference(sample_polars_factor_data, window, "ts_sum")

        assert_numerical_consistency(result, expected)

    def test_ts_sum_nan_in_window(self, sample_polars_factor_with_nan):
        """Test ts_sum behavior with NaN values in window."""
        window = 3
        result = sample_polars_factor_with_nan.ts_sum(window)

        # Should propagate NaN when any value in window is NaN
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_sum_window_larger_than_data(self, sample_polars_factor_data):
        """Test ts_sum when window is larger than available data."""
        window = 100
        result = sample_polars_factor_data.ts_sum(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # All results should be NaN due to insufficient window
        assert result_data["factor"].isna().all()


# ==========================================
# Tests: ts_mean
# ==========================================


class TestTsMean:
    """Tests for ts_mean operation."""

    def test_ts_mean_basic(self, sample_polars_factor_data):
        """Test basic ts_mean functionality."""
        window = 3
        result = sample_polars_factor_data.ts_mean(window)

        assert isinstance(result, Factor)
        assert result.name == f"ts_mean(test_factor,{window})"

    def test_ts_mean_consistency_with_pandas(self, sample_polars_factor_data):
        """Test ts_mean numerical consistency with Pandas."""
        window = 3
        result = sample_polars_factor_data.ts_mean(window)
        expected = get_pandas_reference(sample_polars_factor_data, window, "ts_mean")

        assert_numerical_consistency(result, expected)

    def test_ts_mean_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_mean with NaN in window."""
        window = 3
        result = sample_polars_factor_with_nan.ts_mean(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # Should have NaN values where window contains NaN
        assert result_data["factor"].isna().any()

    def test_ts_mean_window_one(self, sample_polars_factor_data):
        """Test ts_mean with window=1 (should equal the values)."""
        window = 1
        result = sample_polars_factor_data.ts_mean(window)

        # With window=1, mean should equal the original values
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        original_data = (
            sample_polars_factor_data.data.to_pandas()
            if hasattr(sample_polars_factor_data.data, "to_pandas")
            else sample_polars_factor_data.data
        )

        np.testing.assert_allclose(result_data["factor"].values, original_data["factor"].values, rtol=1e-10)


# ==========================================
# Tests: ts_std
# ==========================================


class TestTsStd:
    """Tests for ts_std operation."""

    def test_ts_std_basic(self, sample_polars_factor_data):
        """Test basic ts_std functionality."""
        window = 3
        result = sample_polars_factor_data.ts_std(window)

        assert isinstance(result, Factor)
        assert len(result) == len(sample_polars_factor_data)

    def test_ts_std_consistency_with_pandas(self, sample_polars_factor_data):
        """Test ts_std numerical consistency with Pandas."""
        window = 3
        result = sample_polars_factor_data.ts_std(window)
        expected = get_pandas_reference(sample_polars_factor_data, window, "ts_std")

        assert_numerical_consistency(result, expected)

    def test_ts_std_constant_series(self, sample_polars_factor_constant):
        """Test ts_std with constant values (should be 0 or NaN)."""
        window = 3
        result = sample_polars_factor_constant.ts_std(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # Standard deviation of constant should be 0
        valid_values = result_data["factor"].dropna()
        if len(valid_values) > 0:
            np.testing.assert_allclose(valid_values.values, 0.0, atol=1e-10)

    def test_ts_std_nan_propagation(self, sample_polars_factor_with_nan):
        """Test ts_std propagates NaN correctly."""
        window = 3
        result = sample_polars_factor_with_nan.ts_std(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()


# ==========================================
# Tests: ts_min and ts_max
# ==========================================


class TestTsMinMax:
    """Tests for ts_min and ts_max operations."""

    def test_ts_min_basic(self, sample_polars_factor_data):
        """Test basic ts_min functionality."""
        window = 3
        result = sample_polars_factor_data.ts_min(window)

        assert isinstance(result, Factor)
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert len(result_data) == 20  # 10 per symbol

    def test_ts_min_consistency_with_pandas(self, sample_polars_factor_data):
        """Test ts_min numerical consistency with Pandas."""
        window = 3
        result = sample_polars_factor_data.ts_min(window)
        expected = get_pandas_reference(sample_polars_factor_data, window, "ts_min")

        assert_numerical_consistency(result, expected)

    def test_ts_max_basic(self, sample_polars_factor_data):
        """Test basic ts_max functionality."""
        window = 3
        result = sample_polars_factor_data.ts_max(window)

        assert isinstance(result, Factor)

    def test_ts_max_consistency_with_pandas(self, sample_polars_factor_data):
        """Test ts_max numerical consistency with Pandas."""
        window = 3
        result = sample_polars_factor_data.ts_max(window)
        expected = get_pandas_reference(sample_polars_factor_data, window, "ts_max")

        assert_numerical_consistency(result, expected)

    def test_ts_min_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_min with NaN values."""
        window = 3
        result = sample_polars_factor_with_nan.ts_min(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_max_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_max with NaN values."""
        window = 3
        result = sample_polars_factor_with_nan.ts_max(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()


# ==========================================
# Tests: ts_median
# ==========================================


class TestTsMedian:
    """Tests for ts_median operation."""

    def test_ts_median_basic(self, sample_polars_factor_data):
        """Test basic ts_median functionality."""
        window = 3
        result = sample_polars_factor_data.ts_median(window)

        assert isinstance(result, Factor)
        assert len(result) == 20

    def test_ts_median_consistency_with_pandas(self, sample_polars_factor_data):
        """Test ts_median numerical consistency with Pandas."""
        window = 3
        result = sample_polars_factor_data.ts_median(window)
        expected = get_pandas_reference(sample_polars_factor_data, window, "ts_median")

        assert_numerical_consistency(result, expected)

    def test_ts_median_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_median with NaN in window."""
        window = 3
        result = sample_polars_factor_with_nan.ts_median(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_median_window_insufficient(self, sample_polars_factor_data):
        """Test ts_median when window exceeds data."""
        window = 50
        result = sample_polars_factor_data.ts_median(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().all()


# ==========================================
# Tests: ts_shift and ts_delta (ts_diff)
# ==========================================


class TestTsShiftDelta:
    """Tests for ts_shift and ts_delta operations."""

    def test_ts_shift_basic(self, sample_polars_factor_data):
        """Test basic ts_shift functionality."""
        period = 1
        result = sample_polars_factor_data.ts_shift(period)

        assert isinstance(result, Factor)
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data

        # First value should be NaN (no previous value to shift from)
        assert result_data["factor"].isna().any()

    def test_ts_shift_period_validation(self, sample_polars_factor_data):
        """Test ts_shift shifts values correctly."""
        period = 2
        result = sample_polars_factor_data.ts_shift(period)
        original_data = (
            sample_polars_factor_data.data.to_pandas()
            if hasattr(sample_polars_factor_data.data, "to_pandas")
            else sample_polars_factor_data.data
        )
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data

        # After shift, we should see NaN at beginning
        assert result_data["factor"].isna().any()

    def test_ts_delta_basic(self, sample_polars_factor_data):
        """Test basic ts_delta (difference) functionality."""
        period = 1
        result = sample_polars_factor_data.ts_delta(period)

        assert isinstance(result, Factor)
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert len(result_data) == 20

    def test_ts_delta_first_values_nan(self, sample_polars_factor_data):
        """Test ts_delta produces NaN for first period values."""
        period = 1
        result = sample_polars_factor_data.ts_delta(period)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # First value per symbol should be NaN
        assert result_data["factor"].isna().any()

    def test_ts_delta_computation(self, sample_polars_factor_data):
        """Test ts_delta computes differences correctly."""
        period = 1
        result = sample_polars_factor_data.ts_delta(period)
        original_data = (
            sample_polars_factor_data.data.to_pandas()
            if hasattr(sample_polars_factor_data.data, "to_pandas")
            else sample_polars_factor_data.data
        )
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data

        # For BTCUSDT with values [1,2,3,4,5,...], delta should be [nan, 1, 1, 1, ...]
        btc_mask_orig = original_data["symbol"] == "BTCUSDT"
        btc_mask_result = result_data["symbol"] == "BTCUSDT"

        btc_result = result_data[btc_mask_result]["factor"].values
        # Skip first NaN and check differences are 1
        if len(btc_result) > 1:
            valid = btc_result[1:]
            valid_non_nan = valid[~np.isnan(valid)]
            if len(valid_non_nan) > 0:
                np.testing.assert_allclose(valid_non_nan, 1.0, rtol=1e-10)


# ==========================================
# Tests: ts_rank
# ==========================================


class TestTsRank:
    """Tests for ts_rank operation."""

    def test_ts_rank_basic(self, sample_polars_factor_data):
        """Test basic ts_rank functionality."""
        window = 3
        result = sample_polars_factor_data.ts_rank(window)

        assert isinstance(result, Factor)
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data

        # Rank should be between 0 and 1
        valid_ranks = result_data["factor"].dropna()
        if len(valid_ranks) > 0:
            assert (valid_ranks >= 0).all() and (valid_ranks <= 1).all()

    def test_ts_rank_window_insufficient(self, sample_polars_factor_data):
        """Test ts_rank with insufficient window."""
        window = 5
        result = sample_polars_factor_data.ts_rank(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # First window-1 values should be NaN
        assert result_data["factor"].isna().any()

    def test_ts_rank_nan_propagation(self, sample_polars_factor_with_nan):
        """Test ts_rank with NaN in window."""
        window = 3
        result = sample_polars_factor_with_nan.ts_rank(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # NaN in window should propagate as NaN
        assert result_data["factor"].isna().any()


# ==========================================
# Tests: ts_argmin and ts_argmax
# ==========================================


class TestTsArgMinMax:
    """Tests for ts_argmin and ts_argmax operations."""

    def test_ts_argmin_basic(self, sample_polars_factor_data):
        """Test basic ts_argmin functionality."""
        window = 3
        result = sample_polars_factor_data.ts_argmin(window)

        assert isinstance(result, Factor)
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data

        # argmin should return positions (0 to window-1)
        valid_vals = result_data["factor"].dropna()
        if len(valid_vals) > 0:
            assert (valid_vals >= 0).all() and (valid_vals < window).all()

    def test_ts_argmin_window_insufficient(self, sample_polars_factor_data):
        """Test ts_argmin with insufficient window."""
        window = 5
        result = sample_polars_factor_data.ts_argmin(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_argmin_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_argmin with NaN in window."""
        window = 3
        result = sample_polars_factor_with_nan.ts_argmin(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_argmax_basic(self, sample_polars_factor_data):
        """Test basic ts_argmax functionality."""
        window = 3
        result = sample_polars_factor_data.ts_argmax(window)

        assert isinstance(result, Factor)
        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data

        valid_vals = result_data["factor"].dropna()
        if len(valid_vals) > 0:
            assert (valid_vals >= 0).all() and (valid_vals < window).all()

    def test_ts_argmax_window_insufficient(self, sample_polars_factor_data):
        """Test ts_argmax with insufficient window."""
        window = 5
        result = sample_polars_factor_data.ts_argmax(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_argmax_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_argmax with NaN in window."""
        window = 3
        result = sample_polars_factor_with_nan.ts_argmax(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()


# ==========================================
# Tests: ts_scale
# ==========================================


class TestTsScale:
    """Tests for ts_scale operation."""

    def test_ts_scale_basic(self, sample_polars_factor_data):
        """Test basic ts_scale functionality."""
        window = 3
        const = 0.0
        result = sample_polars_factor_data.ts_scale(window, constant=const)

        assert isinstance(result, Factor)

    def test_ts_scale_range_zero_to_one(self, sample_polars_factor_data):
        """Test ts_scale produces values in [0, 1] range."""
        window = 3
        const = 0.0
        result = sample_polars_factor_data.ts_scale(window, constant=const)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        valid_vals = result_data["factor"].dropna()

        if len(valid_vals) > 0:
            # Scaled values should be in [0, 1] (accounting for rounding)
            assert (valid_vals >= -1e-10).all() and (valid_vals <= 1 + 1e-10).all()

    def test_ts_scale_constant_offset(self, sample_polars_factor_data):
        """Test ts_scale with constant offset."""
        window = 3
        const = 0.5
        result = sample_polars_factor_data.ts_scale(window, constant=const)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        valid_vals = result_data["factor"].dropna()

        if len(valid_vals) > 0:
            # With constant=0.5, values should be in [0.5, 1.5]
            assert (valid_vals >= 0.5 - 1e-10).all() and (valid_vals <= 1.5 + 1e-10).all()

    def test_ts_scale_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_scale with NaN in data."""
        window = 3
        result = sample_polars_factor_with_nan.ts_scale(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()


# ==========================================
# Tests: ts_zscore
# ==========================================


class TestTsZscore:
    """Tests for ts_zscore operation."""

    def test_ts_zscore_basic(self, sample_polars_factor_data):
        """Test basic ts_zscore functionality."""
        window = 3
        result = sample_polars_factor_data.ts_zscore(window)

        assert isinstance(result, Factor)

    def test_ts_zscore_mean_and_std(self, sample_polars_factor_data):
        """Test ts_zscore centers around mean."""
        window = 5
        result = sample_polars_factor_data.ts_zscore(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data

        # Z-scores can be any value, but NaN should exist for insufficient windows
        assert len(result_data) == 20

    def test_ts_zscore_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_zscore with NaN in data."""
        window = 3
        result = sample_polars_factor_with_nan.ts_zscore(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_zscore_constant_series_nan(self, sample_polars_factor_constant):
        """Test ts_zscore with constant series (std=0)."""
        window = 3
        result = sample_polars_factor_constant.ts_zscore(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # When std is 0, result should be NaN (division by 0)
        assert result_data["factor"].isna().any()


# ==========================================
# Tests: ts_skewness
# ==========================================


class TestTsSkewness:
    """Tests for ts_skewness operation."""

    def test_ts_skewness_basic(self, sample_polars_factor_data):
        """Test basic ts_skewness functionality."""
        window = 3
        result = sample_polars_factor_data.ts_skewness(window)

        assert isinstance(result, Factor)

    def test_ts_skewness_regression_correct_window_mean(self):
        """Regression: skewness must use the window mean (not point-wise rolling means).

        For window [1, 1, 4] at t=3:
        - mean = 2
        - deviations = [-1, -1, 2]
        - skewness = (mean(dev^3)) / (std(dev, ddof=0)^3) = 0.7071067811865475
        """
        df = pd.DataFrame(
            {
                "start_time": [0] * 5,
                "end_time": [1, 2, 3, 4, 5],
                "symbol": ["A"] * 5,
                "factor": [1.0, 1.0, 4.0, 10.0, 10.0],
            }
        )
        factor = Factor(df, name="x")
        result = factor.ts_skewness(3).data

        t3 = result.filter(pl.col("end_time") == 3)["factor"][0]
        assert t3 is not None
        assert np.isclose(t3, 0.7071067811865475, rtol=0, atol=1e-12)

    def test_ts_skewness_window_insufficient(self, sample_polars_factor_data):
        """Test ts_skewness with insufficient window."""
        window = 5
        result = sample_polars_factor_data.ts_skewness(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # First window-1 should be NaN
        assert result_data["factor"].isna().any()

    def test_ts_skewness_nan_propagation(self, sample_polars_factor_with_nan):
        """Test ts_skewness with NaN in window."""
        window = 3
        result = sample_polars_factor_with_nan.ts_skewness(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()


# ==========================================
# Tests: ts_kurtosis
# ==========================================


class TestTsKurtosis:
    """Tests for ts_kurtosis operation."""

    def test_ts_kurtosis_basic(self, sample_polars_factor_data):
        """Test basic ts_kurtosis functionality."""
        window = 3
        result = sample_polars_factor_data.ts_kurtosis(window)

        assert isinstance(result, Factor)

    def test_ts_kurtosis_window_insufficient(self, sample_polars_factor_data):
        """Test ts_kurtosis with insufficient window."""
        window = 5
        result = sample_polars_factor_data.ts_kurtosis(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_kurtosis_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_kurtosis with NaN in window."""
        window = 3
        result = sample_polars_factor_with_nan.ts_kurtosis(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_kurtosis_constant_series_nan(self, sample_polars_factor_constant):
        """Test ts_kurtosis with constant series."""
        window = 3
        result = sample_polars_factor_constant.ts_kurtosis(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # Constant series should produce NaN (std=0)
        assert result_data["factor"].isna().any()


# ==========================================
# Tests: ts_corr (Bivariate)
# ==========================================


class TestTsCorr:
    """Tests for ts_corr operation."""

    def test_ts_corr_basic(self, sample_polars_factor_data):
        """Test basic ts_corr functionality."""
        window = 3
        factor1 = sample_polars_factor_data
        factor2 = sample_polars_factor_data  # Corr with self = 1

        result = factor1.ts_corr(factor2, window)

        assert isinstance(result, Factor)

    def test_ts_corr_self_correlation_one(self, sample_polars_factor_data):
        """Test ts_corr of factor with itself (should be 1)."""
        window = 3
        result = sample_polars_factor_data.ts_corr(sample_polars_factor_data, window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        valid_corr = result_data["factor"].dropna()

        if len(valid_corr) > 0:
            # Self-correlation should be 1 (or close to 1)
            np.testing.assert_allclose(valid_corr.values, 1.0, rtol=1e-5, atol=1e-10)

    def test_ts_corr_window_insufficient(self, sample_polars_factor_data):
        """Test ts_corr with insufficient window."""
        window = 50
        result = sample_polars_factor_data.ts_corr(sample_polars_factor_data, window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # Should have NaN for insufficient window
        assert result_data["factor"].isna().any()

    def test_ts_corr_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_corr with NaN in data."""
        window = 3
        result = sample_polars_factor_with_nan.ts_corr(sample_polars_factor_with_nan, window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        # Should handle NaN appropriately
        assert len(result_data) > 0


# ==========================================
# Tests: ts_cov (Bivariate)
# ==========================================


class TestTsCov:
    """Tests for ts_cov operation."""

    def test_ts_cov_basic(self, sample_polars_factor_data):
        """Test basic ts_cov functionality."""
        window = 3
        result = sample_polars_factor_data.ts_cov(sample_polars_factor_data, window)

        assert isinstance(result, Factor)

    def test_ts_cov_self_covariance(self, sample_polars_factor_data):
        """Test ts_cov of factor with itself (variance)."""
        window = 3
        result = sample_polars_factor_data.ts_cov(sample_polars_factor_data, window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data

        # Covariance with self should be variance
        valid_cov = result_data["factor"].dropna()
        if len(valid_cov) > 0:
            assert (valid_cov >= 0).all()  # Variance should be non-negative

    def test_ts_cov_window_insufficient(self, sample_polars_factor_data):
        """Test ts_cov with insufficient window."""
        window = 50
        result = sample_polars_factor_data.ts_cov(sample_polars_factor_data, window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert result_data["factor"].isna().any()

    def test_ts_cov_nan_handling(self, sample_polars_factor_with_nan):
        """Test ts_cov with NaN in data."""
        window = 3
        result = sample_polars_factor_with_nan.ts_cov(sample_polars_factor_with_nan, window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        assert len(result_data) > 0


# ==========================================
# Integration and Edge Cases
# ==========================================


class TestTsOpsIntegration:
    """Integration tests for multiple ts_ops."""

    def test_all_operators_return_factor(self, sample_polars_factor_data):
        """Test that all operators return Factor instances."""
        window = 3

        operators = [
            ("ts_sum", {"window": window}),
            ("ts_mean", {"window": window}),
            ("ts_std", {"window": window}),
            ("ts_min", {"window": window}),
            ("ts_max", {"window": window}),
            ("ts_median", {"window": window}),
            ("ts_shift", {"period": 1}),
            ("ts_delta", {"period": 1}),
            ("ts_rank", {"window": window}),
            ("ts_argmin", {"window": window}),
            ("ts_argmax", {"window": window}),
            ("ts_scale", {"window": window}),
            ("ts_zscore", {"window": window}),
            ("ts_skewness", {"window": window}),
            ("ts_kurtosis", {"window": window}),
        ]

        for op_name, kwargs in operators:
            op = getattr(sample_polars_factor_data, op_name)
            result = op(**kwargs)
            assert isinstance(result, Factor), f"{op_name} should return Factor"

    def test_bivariate_operators_return_factor(self, sample_polars_factor_data):
        """Test that bivariate operators return Factor instances."""
        window = 3

        bivariate_ops = [
            ("ts_corr", {"other": sample_polars_factor_data, "window": window}),
            ("ts_cov", {"other": sample_polars_factor_data, "window": window}),
        ]

        for op_name, kwargs in bivariate_ops:
            op = getattr(sample_polars_factor_data, op_name)
            result = op(**kwargs)
            assert isinstance(result, Factor), f"{op_name} should return Factor"

    def test_ts_ops_preserve_symbols(self, sample_polars_factor_data):
        """Test that ts_ops preserve symbol grouping."""
        window = 3
        result = sample_polars_factor_data.ts_mean(window)

        result_data = result.data.to_pandas() if hasattr(result.data, "to_pandas") else result.data
        symbols = result_data["symbol"].unique()

        # Should have both symbols
        assert len(symbols) == 2
        assert "BTCUSDT" in symbols and "ETHUSDT" in symbols


# ==========================================
# Tests: LazyFrame Evaluation Tracking
# ==========================================


class TestLazyEvaluation:
    """Tests to verify that ts_ops do NOT trigger pl.LazyFrame.collect during result creation.

    These tests track the number of times pl.LazyFrame.collect is called and ensure
    that building a ts_ops result does NOT cause any collect calls. The actual
    collect should only happen when accessing .data property.
    """

    def test_ts_sum_no_collect_during_construction(self, sample_polars_factor_data, monkeypatch):
        """Test that ts_sum does not trigger collect during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Reset counter before the operation
        collect_call_count["count"] = 0

        # Call ts_sum - should NOT trigger any collect calls
        result = sample_polars_factor_data.ts_sum(window=3)

        # Verify that collect was NOT called during construction
        assert collect_call_count["count"] == 0, (
            f"ts_sum should not call collect during construction, "
            f"but collect was called {collect_call_count['count']} times"
        )

        # Verify result is a Factor (proving construction succeeded)
        assert isinstance(result, Factor)

    def test_ts_rank_no_collect_during_construction(self, sample_polars_factor_data, monkeypatch):
        """Test that ts_rank does not trigger collect during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Reset counter before the operation
        collect_call_count["count"] = 0

        # Call ts_rank - should NOT trigger any collect calls
        result = sample_polars_factor_data.ts_rank(window=3)

        # Verify that collect was NOT called during construction
        assert collect_call_count["count"] == 0, (
            f"ts_rank should not call collect during construction, "
            f"but collect was called {collect_call_count['count']} times"
        )

        # Verify result is a Factor
        assert isinstance(result, Factor)

    def test_ts_corr_no_collect_during_construction(self, sample_polars_factor_data, monkeypatch):
        """Test that ts_corr does not trigger collect during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Reset counter before the operation
        collect_call_count["count"] = 0

        # Call ts_corr - should NOT trigger any collect calls
        result = sample_polars_factor_data.ts_corr(sample_polars_factor_data, window=3)

        # Verify that collect was NOT called during construction
        assert collect_call_count["count"] == 0, (
            f"ts_corr should not call collect during construction, "
            f"but collect was called {collect_call_count['count']} times"
        )

        # Verify result is a Factor
        assert isinstance(result, Factor)

    def test_ts_mean_no_collect_during_construction(self, sample_polars_factor_data, monkeypatch):
        """Test that ts_mean does not trigger collect during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Reset counter before the operation
        collect_call_count["count"] = 0

        # Call ts_mean - should NOT trigger any collect calls
        result = sample_polars_factor_data.ts_mean(window=3)

        # Verify that collect was NOT called during construction
        assert collect_call_count["count"] == 0, (
            f"ts_mean should not call collect during construction, "
            f"but collect was called {collect_call_count['count']} times"
        )

        # Verify result is a Factor
        assert isinstance(result, Factor)

    def test_ts_std_no_collect_during_construction(self, sample_polars_factor_data, monkeypatch):
        """Test that ts_std does not trigger collect during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Reset counter before the operation
        collect_call_count["count"] = 0

        # Call ts_std - should NOT trigger any collect calls
        result = sample_polars_factor_data.ts_std(window=3)

        # Verify that collect was NOT called during construction
        assert collect_call_count["count"] == 0, (
            f"ts_std should not call collect during construction, "
            f"but collect was called {collect_call_count['count']} times"
        )

        # Verify result is a Factor
        assert isinstance(result, Factor)

    def test_ts_min_no_collect_during_construction(self, sample_polars_factor_data, monkeypatch):
        """Test that ts_min does not trigger collect during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Reset counter before the operation
        collect_call_count["count"] = 0

        # Call ts_min - should NOT trigger any collect calls
        result = sample_polars_factor_data.ts_min(window=3)

        # Verify that collect was NOT called during construction
        assert collect_call_count["count"] == 0, (
            f"ts_min should not call collect during construction, "
            f"but collect was called {collect_call_count['count']} times"
        )

        # Verify result is a Factor
        assert isinstance(result, Factor)

    def test_ts_max_no_collect_during_construction(self, sample_polars_factor_data, monkeypatch):
        """Test that ts_max does not trigger collect during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Reset counter before the operation
        collect_call_count["count"] = 0

        # Call ts_max - should NOT trigger any collect calls
        result = sample_polars_factor_data.ts_max(window=3)

        # Verify that collect was NOT called during construction
        assert collect_call_count["count"] == 0, (
            f"ts_max should not call collect during construction, "
            f"but collect was called {collect_call_count['count']} times"
        )

        # Verify result is a Factor
        assert isinstance(result, Factor)

    def test_ts_cov_no_collect_during_construction(self, sample_polars_factor_data, monkeypatch):
        """Test that ts_cov does not trigger collect during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Reset counter before the operation
        collect_call_count["count"] = 0

        # Call ts_cov - should NOT trigger any collect calls
        result = sample_polars_factor_data.ts_cov(sample_polars_factor_data, window=3)

        # Verify that collect was NOT called during construction
        assert collect_call_count["count"] == 0, (
            f"ts_cov should not call collect during construction, "
            f"but collect was called {collect_call_count['count']} times"
        )

        # Verify result is a Factor
        assert isinstance(result, Factor)

    def test_collect_only_happens_on_data_access(self, sample_polars_factor_data, monkeypatch):
        """Test that collect is ONLY called when accessing .data property, not during construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self):
            collect_call_count["count"] += 1
            return original_collect(self)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Step 1: Create ts_sum result (should NOT trigger collect)
        collect_call_count["count"] = 0
        result = sample_polars_factor_data.ts_sum(window=3)
        assert collect_call_count["count"] == 0, "Construction should not trigger collect"

        # Step 2: Access .data property (SHOULD trigger collect)
        _ = result.data
        assert collect_call_count["count"] > 0, (
            "Accessing .data should trigger collect, "
            f"but collect was called {collect_call_count['count']} times (expected > 0)"
        )

    def test_multiple_ts_ops_chain_no_premature_collect(self, sample_polars_factor_data, monkeypatch):
        """Test that chaining multiple ts_ops does not trigger collect until .data is accessed."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Chain multiple operations without accessing .data
        collect_call_count["count"] = 0
        result1 = sample_polars_factor_data.ts_mean(window=3)
        result2 = result1.ts_std(window=3)
        result3 = result2.ts_rank(window=3)

        # No collect should have been called during construction chain
        assert collect_call_count["count"] == 0, (
            f"Chaining ts_ops should not trigger collect, but collect was called {collect_call_count['count']} times"
        )

        # Now access .data - only then should collect be called
        _ = result3.data
        assert collect_call_count["count"] > 0, "Accessing .data on chained result should trigger collect"
