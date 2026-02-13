"""Regression tests for safe_* operations and strict NaN propagation semantics.

These tests lock in the behavior documented in docs/dev/safe-operations.md:
1. Strict NaN propagation (any NaN in window → NaN result)
2. Safe division (near-zero denominator → NaN/null)
3. Window completeness (partial windows → NaN/null)
4. Degenerate-case handling (constant values, zero variance, etc.)

Reference: Issue #10
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factorium.backtest.utils import safe_divide
from factorium.constants import EPSILON
from factorium.factors.core import Factor


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _make_factor(
    values: list[float],
    symbols: list[str] | None = None,
    name: str = "test",
) -> Factor:
    """Create a Factor from a flat list of values for a single or multiple symbols.

    If symbols is None, uses a single symbol "A" for all values.
    If symbols is provided, len(values) must be divisible by len(symbols).
    """
    if symbols is None:
        symbols = ["A"]

    n_symbols = len(symbols)
    n_times = len(values) // n_symbols
    assert len(values) == n_symbols * n_times, "values length must be divisible by number of symbols"

    rows = []
    for t in range(n_times):
        for s_idx, sym in enumerate(symbols):
            rows.append(
                {
                    "start_time": t * 60000,
                    "end_time": (t + 1) * 60000,
                    "symbol": sym,
                    "factor": values[t * n_symbols + s_idx],
                }
            )

    df = pl.DataFrame(rows)
    return Factor(df, name=name)


def _get_values(factor: Factor) -> list[float | None]:
    """Extract factor values as a list (None for null)."""
    return factor.data["factor"].to_list()


def _is_missing(value: float | None) -> bool:
    """Check if a value is missing (None/null or NaN).

    Polars uses null (None) for incomplete windows and NaN for windows
    contaminated by NaN input. Both represent "missing" in our semantics.
    """
    if value is None:
        return True
    try:
        return np.isnan(value)
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════
# 1. safe_divide (backtest.utils)
# ══════════════════════════════════════════════════════════


class TestSafeDivideScalar:
    """Test safe_divide with scalar inputs."""

    def test_normal_division(self):
        assert safe_divide(10.0, 2.0) == 5.0

    def test_zero_denominator(self):
        result = safe_divide(1.0, 0.0)
        assert np.isnan(result)

    def test_near_zero_denominator(self):
        result = safe_divide(1.0, 1e-15)
        assert np.isnan(result)

    def test_epsilon_boundary_below(self):
        """Value at exactly EPSILON should return default."""
        result = safe_divide(1.0, EPSILON)
        assert np.isnan(result)

    def test_epsilon_boundary_above(self):
        """Value above EPSILON should compute normally."""
        result = safe_divide(1.0, EPSILON * 10)
        assert not np.isnan(result)
        assert result == pytest.approx(1.0 / (EPSILON * 10))

    def test_nan_denominator(self):
        result = safe_divide(1.0, np.nan)
        assert np.isnan(result)

    def test_negative_near_zero(self):
        result = safe_divide(1.0, -1e-15)
        assert np.isnan(result)

    def test_custom_default(self):
        result = safe_divide(1.0, 0.0, default=0.0)
        assert result == 0.0

    def test_negative_denominator(self):
        assert safe_divide(10.0, -2.0) == -5.0

    def test_zero_numerator(self):
        assert safe_divide(0.0, 2.0) == 0.0


class TestSafeDivideArray:
    """Test safe_divide with numpy array inputs."""

    def test_normal_array(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        result = safe_divide(a, b)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_zero_in_array(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 0.0, 6.0])
        result = safe_divide(a, b)
        assert result[0] == pytest.approx(0.5)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(0.5)

    def test_nan_in_array(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, np.nan])
        result = safe_divide(a, b)
        assert result[0] == pytest.approx(0.5)
        assert np.isnan(result[1])

    def test_near_zero_in_array(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 1e-15])
        result = safe_divide(a, b)
        assert result[0] == pytest.approx(0.5)
        assert np.isnan(result[1])


class TestSafeDivideSeries:
    """Test safe_divide with pandas Series inputs."""

    def test_normal_series(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([2.0, 4.0, 6.0])
        result = safe_divide(a, b)
        pd.testing.assert_series_equal(result, pd.Series([0.5, 0.5, 0.5]))

    def test_zero_in_series(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([2.0, 0.0, 6.0])
        result = safe_divide(a, b)
        assert result.iloc[0] == pytest.approx(0.5)
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(0.5)

    def test_nan_in_series(self):
        a = pd.Series([1.0, 2.0])
        b = pd.Series([2.0, np.nan])
        result = safe_divide(a, b)
        assert result.iloc[0] == pytest.approx(0.5)
        assert np.isnan(result.iloc[1])

    def test_empty_series(self):
        a = pd.Series(dtype=float)
        b = pd.Series(dtype=float)
        result = safe_divide(a, b)
        assert len(result) == 0


# ══════════════════════════════════════════════════════════
# 2. Factor Division (safe_div via __truediv__)
# ══════════════════════════════════════════════════════════


class TestFactorDivision:
    """Test Factor / Factor and Factor / scalar with EPSILON guard."""

    def test_normal_factor_division(self):
        f1 = _make_factor([10.0, 20.0, 30.0])
        f2 = _make_factor([2.0, 4.0, 5.0])
        result = _get_values(f1 / f2)
        assert result == pytest.approx([5.0, 5.0, 6.0])

    def test_factor_division_by_zero(self):
        f1 = _make_factor([10.0, 20.0])
        f2 = _make_factor([2.0, 0.0])
        result = _get_values(f1 / f2)
        assert result[0] == pytest.approx(5.0)
        assert result[1] is None  # Polars null

    def test_factor_division_by_near_zero(self):
        f1 = _make_factor([10.0, 20.0])
        f2 = _make_factor([2.0, 1e-15])
        result = _get_values(f1 / f2)
        assert result[0] == pytest.approx(5.0)
        assert result[1] is None

    def test_scalar_division_by_zero(self):
        f1 = _make_factor([10.0, 20.0])
        result = _get_values(f1 / 0.0)
        assert all(v is None for v in result)

    def test_scalar_division_by_near_zero(self):
        f1 = _make_factor([10.0, 20.0])
        result = _get_values(f1 / 1e-15)
        assert all(v is None for v in result)

    def test_scalar_division_normal(self):
        f1 = _make_factor([10.0, 20.0])
        result = _get_values(f1 / 2.0)
        assert result == pytest.approx([5.0, 10.0])

    def test_reverse_division(self):
        """Test scalar / Factor (rtruediv)."""
        f1 = _make_factor([2.0, 0.0, 4.0])
        result = _get_values(10.0 / f1)
        assert result[0] == pytest.approx(5.0)
        assert result[1] is None  # near zero
        assert result[2] == pytest.approx(2.5)


class TestFactorInverse:
    """Test MathOpsMixin.inverse()."""

    def test_normal_inverse(self):
        f = _make_factor([2.0, 4.0, 5.0])
        result = _get_values(f.inverse())
        assert result == pytest.approx([0.5, 0.25, 0.2])

    def test_inverse_near_zero(self):
        f = _make_factor([2.0, 0.0, 1e-15])
        result = _get_values(f.inverse())
        assert result[0] == pytest.approx(0.5)
        assert result[1] is None
        assert result[2] is None


# ══════════════════════════════════════════════════════════
# 3. Strict NaN Propagation — Time-Series Operations
# ══════════════════════════════════════════════════════════


class TestTsNanPropagation:
    """Any NaN in window → null output; incomplete window → null output."""

    def test_ts_mean_nan_in_window(self):
        f = _make_factor([1.0, np.nan, 3.0, 4.0, 5.0])
        result = _get_values(f.ts_mean(3))
        # Window [1, NaN, 3] → missing, [NaN, 3, 4] → missing, [3, 4, 5] → 4.0
        assert _is_missing(result[0])  # incomplete window
        assert _is_missing(result[1])  # incomplete window
        assert _is_missing(result[2])  # NaN in window
        assert _is_missing(result[3])  # NaN in window
        assert result[4] == pytest.approx(4.0)

    def test_ts_mean_full_clean_window(self):
        f = _make_factor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _get_values(f.ts_mean(3))
        assert _is_missing(result[0])  # window not full
        assert _is_missing(result[1])  # window not full
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_ts_std_nan_propagation(self):
        f = _make_factor([1.0, 2.0, np.nan, 4.0, 5.0])
        result = _get_values(f.ts_std(3))
        assert _is_missing(result[0])  # incomplete window
        assert _is_missing(result[1])  # incomplete window
        assert _is_missing(result[2])  # NaN in window
        assert _is_missing(result[3])  # NaN in window
        assert _is_missing(result[4])  # [nan,4,5] → NaN still in window

    def test_ts_sum_incomplete_window(self):
        """Partial windows return null."""
        f = _make_factor([1.0, 2.0, 3.0])
        result = _get_values(f.ts_sum(3))
        assert _is_missing(result[0])
        assert _is_missing(result[1])
        assert result[2] == pytest.approx(6.0)

    def test_ts_min_max_nan_propagation(self):
        f = _make_factor([1.0, np.nan, 3.0, 4.0])
        result_min = _get_values(f.ts_min(2))
        result_max = _get_values(f.ts_max(2))
        assert _is_missing(result_min[0])  # incomplete
        assert _is_missing(result_min[1])  # NaN in window
        assert _is_missing(result_min[2])  # NaN in window
        assert result_min[3] == pytest.approx(3.0)
        assert result_max[3] == pytest.approx(4.0)


class TestTsWindowCompleteness:
    """Window must have exactly window_size samples."""

    def test_window_larger_than_data(self):
        f = _make_factor([1.0, 2.0])
        result = _get_values(f.ts_mean(5))
        assert all(_is_missing(v) for v in result)

    def test_window_one(self):
        f = _make_factor([1.0, 2.0, 3.0])
        result = _get_values(f.ts_mean(1))
        assert result == pytest.approx([1.0, 2.0, 3.0])


class TestTsDegenerateCases:
    """Degenerate cases: constant values, zero variance, etc."""

    def test_ts_zscore_constant_values(self):
        """All values identical → std = 0 → null."""
        f = _make_factor([5.0, 5.0, 5.0, 5.0, 5.0])
        result = _get_values(f.ts_zscore(3))
        # std = 0 → should return null for all valid windows
        assert result[2] is None
        assert result[3] is None
        assert result[4] is None

    def test_ts_scale_constant_values(self):
        """All values identical → max-min = 0 → null."""
        f = _make_factor([3.0, 3.0, 3.0, 3.0])
        result = _get_values(f.ts_scale(3))
        assert result[2] is None
        assert result[3] is None

    def test_ts_rank_constant_values(self):
        """All values identical → std < EPSILON → null."""
        f = _make_factor([7.0, 7.0, 7.0, 7.0])
        result = _get_values(f.ts_rank(3))
        assert result[2] is None
        assert result[3] is None

    def test_ts_corr_constant_x(self):
        """One factor is constant → std = 0 → null."""
        f1 = _make_factor([1.0, 2.0, 3.0, 4.0, 5.0])
        f2 = _make_factor([3.0, 3.0, 3.0, 3.0, 3.0])
        result = _get_values(f1.ts_corr(f2, 3))
        # f2 has zero std → correlation is undefined
        assert result[2] is None
        assert result[3] is None
        assert result[4] is None

    def test_ts_beta_zero_variance(self):
        """Regressor has zero variance → beta undefined → null."""
        f1 = _make_factor([1.0, 2.0, 3.0, 4.0])
        f2 = _make_factor([5.0, 5.0, 5.0, 5.0])
        result = _get_values(f1.ts_beta(f2, 3))
        assert result[2] is None
        assert result[3] is None

    def test_ts_corr_minimum_window(self):
        """ts_corr with window < 2 should return all null."""
        f1 = _make_factor([1.0, 2.0, 3.0])
        f2 = _make_factor([4.0, 5.0, 6.0])
        result = _get_values(f1.ts_corr(f2, 1))
        assert all(v is None for v in result)


# ══════════════════════════════════════════════════════════
# 4. Strict NaN Propagation — Cross-Sectional Operations
# ══════════════════════════════════════════════════════════


class TestCsNanPropagation:
    """If ANY symbol has NaN at time t, ALL symbols get null at time t."""

    def test_cs_rank_with_nan(self):
        """One symbol has NaN → entire cross-section is null."""
        f = _make_factor(
            [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
            symbols=["A", "B"],
        )
        result = _get_values(f.cs_rank())
        # t=0: [1, 2] → valid
        assert result[0] is not None
        assert result[1] is not None
        # t=1: [3, NaN] → both null
        assert result[2] is None
        assert result[3] is None
        # t=2: [5, 6] → valid
        assert result[4] is not None
        assert result[5] is not None

    def test_cs_zscore_with_nan(self):
        f = _make_factor(
            [1.0, 2.0, np.nan, 4.0],
            symbols=["A", "B"],
        )
        result = _get_values(f.cs_zscore())
        # t=0: [1, 2] → valid
        assert result[0] is not None
        assert result[1] is not None
        # t=1: [NaN, 4] → both null
        assert result[2] is None
        assert result[3] is None

    def test_cs_demean_with_nan(self):
        f = _make_factor(
            [10.0, np.nan, 30.0, 40.0],
            symbols=["A", "B"],
        )
        result = _get_values(f.cs_demean())
        # t=0: [10, NaN] → both null
        assert result[0] is None
        assert result[1] is None
        # t=1: [30, 40] → valid
        assert result[2] is not None
        assert result[3] is not None

    def test_cs_winsorize_with_nan(self):
        f = _make_factor(
            [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
            symbols=["A", "B", "C", "D"],
        )
        result = _get_values(f.cs_winsorize(0.25))
        # t=0: [1, 2, 3, NaN] → all null
        assert all(v is None for v in result[:4])
        # t=1: [5, 6, 7, 8] → all valid
        assert all(v is not None for v in result[4:])

    def test_cs_rank_all_valid(self):
        f = _make_factor(
            [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
            symbols=["A", "B", "C"],
        )
        result = _get_values(f.cs_rank())
        assert all(v is not None for v in result)


class TestCsDegenerateCases:
    """Cross-sectional degenerate cases."""

    def test_cs_neutralize_constant_regressor(self):
        """Regressor has zero variance → beta undefined → null."""
        f = _make_factor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            symbols=["A", "B", "C"],
        )
        const = _make_factor(
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            symbols=["A", "B", "C"],
        )
        result = _get_values(f.cs_neutralize(const))
        # Constant regressor → var = 0 → null
        assert all(v is None for v in result)


# ══════════════════════════════════════════════════════════
# 5. Math Operations Safety
# ══════════════════════════════════════════════════════════


class TestMathSafety:
    """Test math operations with domain-safety guards."""

    def test_log_positive(self):
        f = _make_factor([1.0, np.e, 10.0])
        result = _get_values(f.log())
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_log_negative(self):
        f = _make_factor([-1.0, 0.0, 1.0])
        result = _get_values(f.log())
        assert result[0] is None  # log of negative
        assert result[1] is None  # log of zero
        assert result[2] == pytest.approx(0.0)

    def test_sqrt_negative(self):
        f = _make_factor([-4.0, 0.0, 4.0])
        result = _get_values(f.sqrt())
        assert result[0] is None  # sqrt of negative
        assert result[1] is None  # sqrt of zero (factor > 0 is strict)
        assert result[2] == pytest.approx(2.0)

    def test_log_with_base(self):
        f = _make_factor([1.0, 10.0, 100.0])
        result = _get_values(f.log(base=10))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════
# 6. Multi-Symbol Consistency
# ══════════════════════════════════════════════════════════


class TestMultiSymbolConsistency:
    """Ensure operations are applied per-symbol for ts_* and per-time for cs_*."""

    def test_ts_mean_per_symbol(self):
        """Each symbol should have independent rolling mean."""
        f = _make_factor(
            [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            symbols=["A", "B"],
        )
        result = f.data.sort(["symbol", "end_time"])
        a_vals = result.filter(pl.col("symbol") == "A")["factor"].to_list()
        b_vals = result.filter(pl.col("symbol") == "B")["factor"].to_list()

        result_mean = f.ts_mean(2).data.sort(["symbol", "end_time"])
        a_mean = result_mean.filter(pl.col("symbol") == "A")["factor"].to_list()
        b_mean = result_mean.filter(pl.col("symbol") == "B")["factor"].to_list()

        assert a_mean[0] is None  # incomplete
        assert a_mean[1] == pytest.approx(1.5)  # mean(1, 2)
        assert a_mean[2] == pytest.approx(2.5)  # mean(2, 3)
        assert b_mean[0] is None
        assert b_mean[1] == pytest.approx(15.0)  # mean(10, 20)
        assert b_mean[2] == pytest.approx(25.0)  # mean(20, 30)

    def test_cs_rank_per_time(self):
        """Cross-sectional rank is computed per time step."""
        f = _make_factor(
            [10.0, 20.0, 30.0, 100.0, 50.0, 1.0],
            symbols=["A", "B", "C"],
        )
        result = f.cs_rank().data.sort(["end_time", "symbol"])
        vals = result["factor"].to_list()

        # t=0: A=10 (rank 1/3), B=20 (rank 2/3), C=30 (rank 3/3)
        assert vals[0] == pytest.approx(1 / 3)
        assert vals[1] == pytest.approx(2 / 3)
        assert vals[2] == pytest.approx(3 / 3)


# ══════════════════════════════════════════════════════════
# 7. EPSILON Threshold Tests
# ══════════════════════════════════════════════════════════


class TestEpsilonThreshold:
    """Test the EPSILON boundary precisely."""

    def test_epsilon_value(self):
        """EPSILON should be 1e-10."""
        assert EPSILON == 1e-10

    def test_division_at_epsilon(self):
        """Division by exactly EPSILON should return null."""
        f1 = _make_factor([1.0])
        f2 = _make_factor([EPSILON])
        result = _get_values(f1 / f2)
        assert result[0] is None

    def test_division_just_above_epsilon(self):
        """Division by value slightly above EPSILON should succeed."""
        f1 = _make_factor([1.0])
        f2 = _make_factor([EPSILON * 2])
        result = _get_values(f1 / f2)
        assert result[0] is not None
        assert result[0] == pytest.approx(1.0 / (EPSILON * 2))

    def test_inverse_at_epsilon(self):
        f = _make_factor([EPSILON])
        result = _get_values(f.inverse())
        assert result[0] is None

    def test_inverse_just_above_epsilon(self):
        f = _make_factor([EPSILON * 2])
        result = _get_values(f.inverse())
        assert result[0] is not None
