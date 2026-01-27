"""Tests for MathOps Polars support - TDD RED phase.

Tests verify that MathOpsMixin operations work correctly with Polars DataFrames,
maintaining numerical consistency with Pandas implementations.

All tests are expected to FAIL until Polars support is fully implemented in MathOpsMixin.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np

from factorium import Factor


@pytest.fixture
def sample_pandas_df():
    """Create sample factor data as pandas DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=20, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6

    data = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for i in range(10):
            data.append(
                {
                    "start_time": int(timestamps[i]),
                    "end_time": int(timestamps[i] + 60000),
                    "symbol": symbol,
                    "factor": float(np.random.randn()),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_polars_df(sample_pandas_df):
    """Create sample factor data as Polars DataFrame."""
    return pl.from_pandas(sample_pandas_df)


@pytest.fixture
def sample_factor_pandas(sample_pandas_df):
    """Create Factor object from pandas DataFrame."""
    return Factor(sample_pandas_df, name="test_factor")


@pytest.fixture
def sample_factor_polars(sample_polars_df):
    """Create Factor object from Polars DataFrame."""
    return Factor(sample_polars_df, name="test_factor_polars")


@pytest.fixture
def positive_factor_pandas():
    """Create factor data with positive values only."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=20, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6

    data = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for i in range(10):
            data.append(
                {
                    "start_time": int(timestamps[i]),
                    "end_time": int(timestamps[i] + 60000),
                    "symbol": symbol,
                    "factor": float(abs(np.random.randn()) + 0.1),  # Ensure positive
                }
            )

    return Factor(pd.DataFrame(data), name="positive_factor")


@pytest.fixture
def positive_factor_polars(positive_factor_pandas):
    """Create positive factor from Polars DataFrame."""
    return Factor(pl.from_pandas(positive_factor_pandas.to_pandas()), name="positive_factor_polars")


@pytest.fixture
def mixed_factor_pandas():
    """Create factor with mixed positive/negative values and some zeros."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=20, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6

    data = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for i in range(10):
            val = np.random.randn()
            if i % 5 == 0:  # Add some zeros
                val = 0.0
            data.append(
                {
                    "start_time": int(timestamps[i]),
                    "end_time": int(timestamps[i] + 60000),
                    "symbol": symbol,
                    "factor": float(val),
                }
            )

    return Factor(pd.DataFrame(data), name="mixed_factor")


@pytest.fixture
def mixed_factor_polars(mixed_factor_pandas):
    """Create mixed factor from Polars DataFrame."""
    return Factor(pl.from_pandas(mixed_factor_pandas.to_pandas()), name="mixed_factor_polars")


class TestMathOpsPolars_Abs:
    """Test abs() operation with Polars."""

    def test_abs_basic_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should compute absolute value correctly with Polars."""
        result_pandas = sample_factor_pandas.abs()
        result_polars = sample_factor_polars.abs()

        # Convert to pandas for comparison
        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Compare factor values
        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_abs_nan_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should preserve NaN values during abs operation."""
        # Add NaN to both
        pandas_df = sample_factor_pandas.to_pandas()
        pandas_df.loc[0, "factor"] = np.nan
        factor_with_nan_pandas = Factor(pandas_df, name="with_nan_pandas")

        polars_df = sample_factor_polars.to_pandas()
        polars_df.loc[0, "factor"] = np.nan
        factor_with_nan_polars = Factor(pl.from_pandas(polars_df), name="with_nan_polars")

        result_pandas = factor_with_nan_pandas.abs()
        result_polars = factor_with_nan_polars.abs()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Check that NaN is preserved at index 0
        assert pd.isna(result_pandas_pd.iloc[0]["factor"])
        assert pd.isna(result_polars_pd.iloc[0]["factor"])


class TestMathOpsPolars_Sign:
    """Test sign() operation with Polars."""

    def test_sign_basic_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should compute sign correctly with Polars."""
        result_pandas = sample_factor_pandas.sign()
        result_polars = sample_factor_polars.sign()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Sign results should be -1, 0, or 1
        assert result_polars_pd["factor"].isin([-1.0, 0.0, 1.0]).all()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_sign_zero_behavior_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should return 0 for zero values."""
        result_pandas = mixed_factor_pandas.sign()
        result_polars = mixed_factor_polars.sign()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Find zero indices
        zero_indices = (mixed_factor_pandas.to_pandas()["factor"] == 0.0).values
        assert (result_polars_pd["factor"].iloc[zero_indices] == 0.0).all()
        assert (result_pandas_pd["factor"].iloc[zero_indices] == 0.0).all()


class TestMathOpsPolars_Inverse:
    """Test inverse() operation with Polars."""

    def test_inverse_basic_behavior_polars(self, positive_factor_polars, positive_factor_pandas):
        """Should compute 1/x correctly with Polars."""
        result_pandas = positive_factor_pandas.inverse()
        result_polars = positive_factor_polars.inverse()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_inverse_zero_behavior_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should return NaN for zero values."""
        result_pandas = mixed_factor_pandas.inverse()
        result_polars = mixed_factor_polars.inverse()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Zero indices should result in NaN
        zero_indices = (mixed_factor_pandas.to_pandas()["factor"] == 0.0).values
        assert result_polars_pd["factor"].iloc[zero_indices].isna().all()
        assert result_pandas_pd["factor"].iloc[zero_indices].isna().all()


class TestMathOpsPolars_Log:
    """Test log() operation with Polars."""

    def test_log_basic_behavior_polars(self, positive_factor_polars, positive_factor_pandas):
        """Should compute natural log correctly with Polars."""
        result_pandas = positive_factor_pandas.log()
        result_polars = positive_factor_polars.log()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_log_nonpositive_behavior_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should return NaN for non-positive values."""
        result_pandas = mixed_factor_pandas.log()
        result_polars = mixed_factor_polars.log()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Non-positive indices should result in NaN
        nonpos_indices = (mixed_factor_pandas.to_pandas()["factor"] <= 0).values
        assert result_polars_pd["factor"].iloc[nonpos_indices].isna().all()
        assert result_pandas_pd["factor"].iloc[nonpos_indices].isna().all()


class TestMathOpsPolars_Ln:
    """Test ln() operation with Polars."""

    def test_ln_basic_behavior_polars(self, positive_factor_polars, positive_factor_pandas):
        """Should compute natural log (same as log()) with Polars."""
        result_pandas = positive_factor_pandas.ln()
        result_polars = positive_factor_polars.ln()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_ln_consistency_with_log_polars(self, positive_factor_polars):
        """ln() should be identical to log()."""
        result_ln = positive_factor_polars.ln()
        result_log = positive_factor_polars.log()

        result_ln_pd = result_ln.to_pandas()
        result_log_pd = result_log.to_pandas()

        np.testing.assert_allclose(
            result_ln_pd["factor"],
            result_log_pd["factor"],
            rtol=1e-14,
            equal_nan=True,
        )


class TestMathOpsPolars_Sqrt:
    """Test sqrt() operation with Polars."""

    def test_sqrt_basic_behavior_polars(self, positive_factor_polars, positive_factor_pandas):
        """Should compute square root correctly with Polars."""
        result_pandas = positive_factor_pandas.sqrt()
        result_polars = positive_factor_polars.sqrt()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_sqrt_negative_behavior_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should return NaN for negative values."""
        result_pandas = mixed_factor_pandas.sqrt()
        result_polars = mixed_factor_polars.sqrt()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Negative indices should result in NaN
        neg_indices = (mixed_factor_pandas.to_pandas()["factor"] < 0).values
        assert result_polars_pd["factor"].iloc[neg_indices].isna().all()
        assert result_pandas_pd["factor"].iloc[neg_indices].isna().all()


class TestMathOpsPolars_SignedLog1p:
    """Test signed_log1p() operation with Polars."""

    def test_signed_log1p_basic_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should compute sign(x) * log1p(|x|) correctly with Polars."""
        result_pandas = sample_factor_pandas.signed_log1p()
        result_polars = sample_factor_polars.signed_log1p()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_signed_log1p_sign_preservation_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should preserve sign of input values."""
        result_pandas = mixed_factor_pandas.signed_log1p()
        result_polars = mixed_factor_polars.signed_log1p()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()
        original_pd = mixed_factor_pandas.to_pandas()

        # Check sign preservation for non-zero values
        nonzero_indices = (original_pd["factor"] != 0).values
        result_signs_polars = np.sign(result_polars_pd["factor"].iloc[nonzero_indices])
        original_signs = np.sign(original_pd["factor"].iloc[nonzero_indices])

        np.testing.assert_array_equal(result_signs_polars, original_signs)


class TestMathOpsPolars_SignedPow:
    """Test signed_pow() operation with Polars."""

    def test_signed_pow_scalar_basic_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should compute sign(x) * |x|^exp correctly with scalar exponent."""
        result_pandas = mixed_factor_pandas.signed_pow(2.0)
        result_polars = mixed_factor_polars.signed_pow(2.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_signed_pow_zero_exponent_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should return 0 for non-zero values with exponent 0.5."""
        result_pandas = mixed_factor_pandas.signed_pow(0.5)
        result_polars = mixed_factor_polars.signed_pow(0.5)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Results should match
        np.testing.assert_allclose(
            result_polars_pd["factor"].fillna(-999),
            result_pandas_pd["factor"].fillna(-999),
            rtol=1e-14,
            equal_nan=True,
        )


class TestMathOpsPolars_Pow:
    """Test pow() operation with Polars."""

    def test_pow_scalar_basic_polars(self, positive_factor_polars, positive_factor_pandas):
        """Should compute x^exp correctly with scalar exponent."""
        result_pandas = positive_factor_pandas.pow(2.0)
        result_polars = positive_factor_polars.pow(2.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_pow_zero_base_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should handle zero base correctly."""
        result_pandas = mixed_factor_pandas.pow(2.0)
        result_polars = mixed_factor_polars.pow(2.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Zero bases should result in 0 for positive exponents
        zero_indices = (mixed_factor_pandas.to_pandas()["factor"] == 0.0).values
        assert (result_polars_pd["factor"].iloc[zero_indices] == 0.0).all()
        assert (result_pandas_pd["factor"].iloc[zero_indices] == 0.0).all()


class TestMathOpsPolars_Where:
    """Test where() operation with Polars."""

    def test_where_scalar_basic_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should select values based on condition."""
        # Create a condition factor
        cond_data = sample_factor_pandas.to_pandas().copy()
        cond_data["factor"] = (cond_data["factor"] > 0).astype(float)
        cond_factor_pandas = Factor(cond_data, name="condition")
        cond_factor_polars = Factor(pl.from_pandas(cond_data), name="condition")

        result_pandas = sample_factor_pandas.where(cond_factor_pandas, other=-999.0)
        result_polars = sample_factor_polars.where(cond_factor_polars, other=-999.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].fillna(-999),
            result_pandas_pd["factor"].fillna(-999),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_where_nan_condition_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should treat NaN in condition as False."""
        cond_data = sample_factor_pandas.to_pandas().copy()
        cond_data.loc[0, "factor"] = np.nan
        cond_factor_pandas = Factor(cond_data, name="condition")
        cond_factor_polars = Factor(pl.from_pandas(cond_data), name="condition")

        result_pandas = sample_factor_pandas.where(cond_factor_pandas, other=0.0)
        result_polars = sample_factor_polars.where(cond_factor_polars, other=0.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # NaN condition should result in "other" value
        assert result_polars_pd.iloc[0]["factor"] == 0.0
        assert result_pandas_pd.iloc[0]["factor"] == 0.0


class TestMathOpsPolars_Max:
    """Test max() operation with Polars."""

    def test_max_scalar_basic_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should compute element-wise maximum with scalar."""
        result_pandas = mixed_factor_pandas.max(0.5)
        result_polars = mixed_factor_polars.max(0.5)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_max_nan_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should handle NaN correctly in max operation."""
        # Add NaN
        pandas_df = sample_factor_pandas.to_pandas()
        pandas_df.loc[0, "factor"] = np.nan
        factor_with_nan_pandas = Factor(pandas_df, name="with_nan")
        factor_with_nan_polars = Factor(pl.from_pandas(pandas_df), name="with_nan")

        result_pandas = factor_with_nan_pandas.max(1.0)
        result_polars = factor_with_nan_polars.max(1.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Compare non-NaN results
        mask = ~result_pandas_pd["factor"].isna()
        np.testing.assert_allclose(
            result_polars_pd["factor"][mask],
            result_pandas_pd["factor"][mask],
            rtol=1e-14,
        )


class TestMathOpsPolars_Min:
    """Test min() operation with Polars."""

    def test_min_scalar_basic_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should compute element-wise minimum with scalar."""
        result_pandas = mixed_factor_pandas.min(0.5)
        result_polars = mixed_factor_polars.min(0.5)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_min_nan_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should handle NaN correctly in min operation."""
        pandas_df = sample_factor_pandas.to_pandas()
        pandas_df.loc[0, "factor"] = np.nan
        factor_with_nan_pandas = Factor(pandas_df, name="with_nan")
        factor_with_nan_polars = Factor(pl.from_pandas(pandas_df), name="with_nan")

        result_pandas = factor_with_nan_pandas.min(1.0)
        result_polars = factor_with_nan_polars.min(1.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        # Compare non-NaN results
        mask = ~result_pandas_pd["factor"].isna()
        np.testing.assert_allclose(
            result_polars_pd["factor"][mask],
            result_pandas_pd["factor"][mask],
            rtol=1e-14,
        )


class TestMathOpsPolars_Add:
    """Test add() operation with Polars."""

    def test_add_scalar_basic_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should add scalar to all values."""
        result_pandas = sample_factor_pandas.add(1.5)
        result_polars = sample_factor_polars.add(1.5)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_add_nan_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should preserve NaN when adding."""
        pandas_df = sample_factor_pandas.to_pandas()
        pandas_df.loc[0, "factor"] = np.nan
        factor_with_nan_pandas = Factor(pandas_df, name="with_nan")
        factor_with_nan_polars = Factor(pl.from_pandas(pandas_df), name="with_nan")

        result_pandas = factor_with_nan_pandas.add(1.0)
        result_polars = factor_with_nan_polars.add(1.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        assert pd.isna(result_polars_pd.iloc[0]["factor"])
        assert pd.isna(result_pandas_pd.iloc[0]["factor"])


class TestMathOpsPolars_Sub:
    """Test sub() operation with Polars."""

    def test_sub_scalar_basic_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should subtract scalar from all values."""
        result_pandas = sample_factor_pandas.sub(1.5)
        result_polars = sample_factor_polars.sub(1.5)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_sub_nan_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should preserve NaN when subtracting."""
        pandas_df = sample_factor_pandas.to_pandas()
        pandas_df.loc[0, "factor"] = np.nan
        factor_with_nan_pandas = Factor(pandas_df, name="with_nan")
        factor_with_nan_polars = Factor(pl.from_pandas(pandas_df), name="with_nan")

        result_pandas = factor_with_nan_pandas.sub(1.0)
        result_polars = factor_with_nan_polars.sub(1.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        assert pd.isna(result_polars_pd.iloc[0]["factor"])
        assert pd.isna(result_pandas_pd.iloc[0]["factor"])


class TestMathOpsPolars_Mul:
    """Test mul() operation with Polars."""

    def test_mul_scalar_basic_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should multiply all values by scalar."""
        result_pandas = sample_factor_pandas.mul(2.0)
        result_polars = sample_factor_polars.mul(2.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_mul_zero_behavior_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should return 0 when multiplying zero by any value."""
        result_pandas = mixed_factor_pandas.mul(3.14)
        result_polars = mixed_factor_polars.mul(3.14)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        zero_indices = (mixed_factor_pandas.to_pandas()["factor"] == 0.0).values
        assert (result_polars_pd["factor"].iloc[zero_indices] == 0.0).all()
        assert (result_pandas_pd["factor"].iloc[zero_indices] == 0.0).all()


class TestMathOpsPolars_Div:
    """Test div() operation with Polars."""

    def test_div_scalar_basic_polars(self, positive_factor_polars, positive_factor_pandas):
        """Should divide all values by scalar."""
        result_pandas = positive_factor_pandas.div(2.0)
        result_polars = positive_factor_polars.div(2.0)

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_div_zero_divisor_polars(self, positive_factor_polars):
        """Should handle division by zero gracefully."""
        # This should not crash
        result_polars = positive_factor_polars.div(0.0)
        result_pd = result_polars.to_pandas()

        # Result should have inf values
        assert np.isinf(result_pd["factor"]).any() or result_pd["factor"].isna().any()


class TestMathOpsPolars_Reverse:
    """Test reverse() operation with Polars."""

    def test_reverse_basic_behavior_polars(self, sample_factor_polars, sample_factor_pandas):
        """Should negate all values."""
        result_pandas = sample_factor_pandas.reverse()
        result_polars = sample_factor_polars.reverse()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        np.testing.assert_allclose(
            result_polars_pd["factor"].sort_values().reset_index(drop=True),
            result_pandas_pd["factor"].sort_values().reset_index(drop=True),
            rtol=1e-14,
            equal_nan=True,
        )

    def test_reverse_zero_behavior_polars(self, mixed_factor_polars, mixed_factor_pandas):
        """Should preserve zero when reversing."""
        result_pandas = mixed_factor_pandas.reverse()
        result_polars = mixed_factor_polars.reverse()

        result_polars_pd = result_polars.to_pandas()
        result_pandas_pd = result_pandas.to_pandas()

        zero_indices = (mixed_factor_pandas.to_pandas()["factor"] == 0.0).values
        assert (result_polars_pd["factor"].iloc[zero_indices] == 0.0).all()
        assert (result_pandas_pd["factor"].iloc[zero_indices] == 0.0).all()


class TestMathOpsPolars_LazyEvaluation:
    """Test that math_ops operations maintain lazy evaluation without premature collect()."""

    def test_abs_does_not_trigger_collect_on_lazy_frame(self, monkeypatch, sample_polars_df):
        """abs() should not call collect() when operating on LazyFrame."""
        # Create LazyFrame directly
        lazy_frame = sample_polars_df.lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        # Track collect calls
        collect_call_count = 0
        original_collect = pl.LazyFrame.collect

        def mock_collect(self, *args, **kwargs):
            nonlocal collect_call_count
            collect_call_count += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", mock_collect)

        # Call abs() - should NOT trigger collect
        result = factor.abs()

        # Verify collect was NOT called during abs() execution
        # (It may be called when creating the factor, but not during abs())
        assert collect_call_count == 0, f"abs() triggered {collect_call_count} collect() calls, expected 0"

    def test_log_does_not_trigger_collect_on_lazy_frame(self, monkeypatch, positive_factor_polars):
        """log() should not call collect() when operating on LazyFrame."""
        # Get the underlying lazy frame if it exists
        polars_df = positive_factor_polars.to_pandas()
        lazy_frame = pl.from_pandas(polars_df).lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        collect_call_count = 0
        original_collect = pl.LazyFrame.collect

        def mock_collect(self, *args, **kwargs):
            nonlocal collect_call_count
            collect_call_count += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", mock_collect)

        # Call log() - should NOT trigger collect
        result = factor.log()

        assert collect_call_count == 0, f"log() triggered {collect_call_count} collect() calls, expected 0"

    def test_where_does_not_trigger_collect_on_lazy_frame(self, monkeypatch, sample_polars_df):
        """where() should not call collect() when operating on LazyFrame."""
        lazy_frame = sample_polars_df.lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        # Create condition factor
        cond_data = sample_polars_df.to_pandas().copy()
        cond_data["factor"] = (cond_data["factor"] > 0).astype(float)
        cond_lazy = pl.from_pandas(cond_data).lazy()
        cond_factor = Factor(cond_lazy, name="condition")

        collect_call_count = 0
        original_collect = pl.LazyFrame.collect

        def mock_collect(self, *args, **kwargs):
            nonlocal collect_call_count
            collect_call_count += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", mock_collect)

        # Call where() - should NOT trigger collect
        result = factor.where(cond_factor, other=-999.0)

        assert collect_call_count == 0, f"where() triggered {collect_call_count} collect() calls, expected 0"

    def test_max_scalar_does_not_trigger_collect_on_lazy_frame(self, monkeypatch, sample_polars_df):
        """max(scalar) should not call collect() when operating on LazyFrame."""
        lazy_frame = sample_polars_df.lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        collect_call_count = 0
        original_collect = pl.LazyFrame.collect

        def mock_collect(self, *args, **kwargs):
            nonlocal collect_call_count
            collect_call_count += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", mock_collect)

        # Call max(scalar) - should NOT trigger collect
        result = factor.max(0.5)

        assert collect_call_count == 0, f"max(scalar) triggered {collect_call_count} collect() calls, expected 0"

    def test_max_factor_scalar_does_not_trigger_collect_on_lazy_frame(self, monkeypatch, sample_polars_df):
        """max(factor, scalar) should not call collect() when operating on LazyFrame."""
        lazy_frame = sample_polars_df.lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        collect_call_count = 0
        original_collect = pl.LazyFrame.collect

        def mock_collect(self, *args, **kwargs):
            nonlocal collect_call_count
            collect_call_count += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", mock_collect)

        # Create second factor for comparison
        other_data = sample_polars_df.to_pandas().copy()
        other_data["factor"] = other_data["factor"] * 0.5
        other_lazy = pl.from_pandas(other_data).lazy()
        other_factor = Factor(other_lazy, name="other_lazy")

        # Reset count before actual operation (initialization may call collect)
        collect_call_count = 0

        # Call max(factor, other_factor) - should NOT trigger collect
        result = factor.max(other_factor)

        assert collect_call_count == 0, (
            f"max(factor, factor) triggered {collect_call_count} collect() calls, expected 0"
        )

    def test_abs_on_lazy_frame_returns_lazy_result(self, sample_polars_df):
        """abs() on LazyFrame should return Factor containing LazyFrame (lazy result)."""
        lazy_frame = sample_polars_df.lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        result = factor.abs()

        # Result should have a LazyFrame (not eager)
        assert hasattr(result, "_lf"), "Result should have _lf attribute"
        assert isinstance(result._lf, pl.LazyFrame), f"Result._lf should be LazyFrame, got {type(result._lf)}"

    def test_log_on_lazy_frame_returns_lazy_result(self, positive_factor_polars):
        """log() on LazyFrame should return Factor containing LazyFrame (lazy result)."""
        polars_df = positive_factor_polars.to_pandas()
        lazy_frame = pl.from_pandas(polars_df).lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        result = factor.log()

        # Result should have a LazyFrame
        assert hasattr(result, "_lf"), "Result should have _lf attribute"
        assert isinstance(result._lf, pl.LazyFrame), f"Result._lf should be LazyFrame, got {type(result._lf)}"

    def test_where_on_lazy_frame_returns_lazy_result(self, sample_polars_df):
        """where() on LazyFrame should return Factor containing LazyFrame (lazy result)."""
        lazy_frame = sample_polars_df.lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        cond_data = sample_polars_df.to_pandas().copy()
        cond_data["factor"] = (cond_data["factor"] > 0).astype(float)
        cond_lazy = pl.from_pandas(cond_data).lazy()
        cond_factor = Factor(cond_lazy, name="condition")

        result = factor.where(cond_factor, other=-999.0)

        # Result should have a LazyFrame
        assert hasattr(result, "_lf"), "Result should have _lf attribute"
        assert isinstance(result._lf, pl.LazyFrame), f"Result._lf should be LazyFrame, got {type(result._lf)}"

    def test_max_on_lazy_frame_returns_lazy_result(self, sample_polars_df):
        """max() on LazyFrame should return Factor containing LazyFrame (lazy result)."""
        lazy_frame = sample_polars_df.lazy()
        factor = Factor(lazy_frame, name="test_lazy")

        result = factor.max(0.5)

        # Result should have a LazyFrame
        assert hasattr(result, "_lf"), "Result should have _lf attribute"
        assert isinstance(result._lf, pl.LazyFrame), f"Result._lf should be LazyFrame, got {type(result._lf)}"
