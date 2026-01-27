"""Tests for CrossSectional Operations with Polars support - TDD RED phase.

Tests cs_rank, cs_zscore, cs_demean, cs_winsorize, cs_neutralize, mean, and median
operations with Polars DataFrames, ensuring strict NaN propagation and consistency
with Pandas implementations.

"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from factorium import Factor


@pytest.fixture
def sample_data_minimal():
    """Create minimal sample factor data (3 symbols, 1 time period)."""
    data = pd.DataFrame(
        [
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 1.0},
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": 2.0},
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 3.0},
        ]
    )
    return data


@pytest.fixture
def sample_data_multi_time():
    """Create sample data with multiple time periods (3 symbols x 2 times)."""
    data = pd.DataFrame(
        [
            # First time period
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 1.0},
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": 2.0},
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 3.0},
            # Second time period
            {"start_time": 1609545600000, "end_time": 1609545600000, "symbol": "A", "factor": 4.0},
            {"start_time": 1609545600000, "end_time": 1609545600000, "symbol": "B", "factor": 5.0},
            {"start_time": 1609545600000, "end_time": 1609545600000, "symbol": "C", "factor": 6.0},
        ]
    )
    return data


@pytest.fixture
def sample_data_with_nan():
    """Create sample data with NaN values in same cross-section."""
    data = pd.DataFrame(
        [
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 1.0},
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": np.nan},
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 3.0},
        ]
    )
    return data


@pytest.fixture
def sample_data_with_nan_multi_time():
    """Create multi-period data with NaN in one time period."""
    data = pd.DataFrame(
        [
            # First time period - clean
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 1.0},
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": 2.0},
            {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 3.0},
            # Second time period - with NaN
            {"start_time": 1609545600000, "end_time": 1609545600000, "symbol": "A", "factor": 4.0},
            {"start_time": 1609545600000, "end_time": 1609545600000, "symbol": "B", "factor": np.nan},
            {"start_time": 1609545600000, "end_time": 1609545600000, "symbol": "C", "factor": 6.0},
        ]
    )
    return data


class TestCSRank:
    """Test cs_rank operation with Polars support."""

    def test_cs_rank_basic_behavior(self, sample_data_minimal):
        """Test basic cs_rank computation on clean data."""
        factor_pd = Factor(sample_data_minimal, name="test_cs_rank")
        ranked = factor_pd.cs_rank()

        # Expected ranks: 1/3, 2/3, 3/3 for values 1.0, 2.0, 3.0
        expected = np.array([1.0 / 3, 2.0 / 3, 1.0])
        result_values = ranked.data.to_pandas()["factor"].values

        np.testing.assert_allclose(result_values, expected, rtol=1e-9, atol=1e-12)

    def test_cs_rank_nan_propagation(self, sample_data_with_nan):
        """Test that cs_rank returns all NaN when any input is NaN (strict mode)."""
        factor_pd = Factor(sample_data_with_nan, name="test_cs_rank_nan")
        ranked = factor_pd.cs_rank()

        # All values should be NaN due to strict NaN propagation
        result_data = ranked.data.to_pandas()
        assert result_data["factor"].isna().all()


class TestCSZscore:
    """Test cs_zscore operation with Polars support."""

    def test_cs_zscore_basic_behavior(self, sample_data_minimal):
        """Test basic cs_zscore computation on clean data."""
        factor_pd = Factor(sample_data_minimal, name="test_cs_zscore")
        zscored = factor_pd.cs_zscore()

        # Values: 1, 2, 3. Mean=2, Std=1. Z-scores = [-1, 0, 1]
        expected = np.array([-1.0, 0.0, 1.0])
        result_values = zscored.data.to_pandas()["factor"].values

        np.testing.assert_allclose(result_values, expected, rtol=1e-9, atol=1e-12)

    def test_cs_zscore_nan_propagation(self, sample_data_with_nan):
        """Test that cs_zscore returns all NaN when any input is NaN (strict mode)."""
        factor_pd = Factor(sample_data_with_nan, name="test_cs_zscore_nan")
        zscored = factor_pd.cs_zscore()

        # All values should be NaN due to strict NaN propagation
        result_data = zscored.data.to_pandas()
        assert result_data["factor"].isna().all()


class TestCSDemean:
    """Test cs_demean operation with Polars support."""

    def test_cs_demean_basic_behavior(self, sample_data_minimal):
        """Test basic cs_demean computation on clean data."""
        factor_pd = Factor(sample_data_minimal, name="test_cs_demean")
        demeaned = factor_pd.cs_demean()

        # Values: 1, 2, 3. Mean=2. Demeaned = [-1, 0, 1]
        expected = np.array([-1.0, 0.0, 1.0])
        result_values = demeaned.data.to_pandas()["factor"].values

        np.testing.assert_allclose(result_values, expected, rtol=1e-9, atol=1e-12)

    def test_cs_demean_nan_propagation(self, sample_data_with_nan):
        """Test that cs_demean returns all NaN when any input is NaN (strict mode)."""
        factor_pd = Factor(sample_data_with_nan, name="test_cs_demean_nan")
        demeaned = factor_pd.cs_demean()

        # All values should be NaN due to strict NaN propagation
        result_data = demeaned.data.to_pandas()
        assert result_data["factor"].isna().all()


class TestCSWinsorize:
    """Test cs_winsorize operation with Polars support."""

    def test_cs_winsorize_basic_behavior(self):
        """Test basic cs_winsorize computation with outliers."""
        data = pd.DataFrame(
            [
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 1.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": 2.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 3.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "D", "factor": 100.0},
            ]
        )
        factor = Factor(data, name="test_cs_winsorize")
        winsorized = factor.cs_winsorize(limits=0.25)

        result_values = winsorized.data.to_pandas()["factor"].values
        # The extreme value (100.0) should be clipped
        assert result_values[3] < 100.0
        # Regular values should remain similar
        assert result_values[0] > 0.0

    def test_cs_winsorize_nan_propagation(self, sample_data_with_nan):
        """Test that cs_winsorize returns all NaN when any input is NaN (strict mode)."""
        factor_pd = Factor(sample_data_with_nan, name="test_cs_winsorize_nan")
        winsorized = factor_pd.cs_winsorize(limits=0.025)

        # All values should be NaN due to strict NaN propagation
        result_data = winsorized.data.to_pandas()
        assert result_data["factor"].isna().all()


class TestCSNeutralize:
    """Test cs_neutralize operation with Polars support."""

    def test_cs_neutralize_basic_behavior(self):
        """Test basic cs_neutralize: Y = 2*X + 1, should get near-zero residuals."""
        # Y = 2*X + 1
        data_y = pd.DataFrame(
            [
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 3.0},  # X=1
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": 5.0},  # X=2
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 7.0},  # X=3
            ]
        )
        data_x = pd.DataFrame(
            [
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 1.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": 2.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 3.0},
            ]
        )

        factor_y = Factor(data_y, name="Y")
        factor_x = Factor(data_x, name="X")

        neutralized = factor_y.cs_neutralize(factor_x)
        result_values = neutralized.data.to_pandas()["factor"].values

        # Should be close to 0 residuals (perfect linear relationship)
        np.testing.assert_allclose(result_values, 0.0, rtol=1e-9, atol=1e-12)

    def test_cs_neutralize_nan_propagation(self, sample_data_with_nan):
        """Test that cs_neutralize returns all NaN when any input is NaN (strict mode)."""
        # Create corresponding X factor with valid data
        data_x = pd.DataFrame(
            [
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 1.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": 2.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 3.0},
            ]
        )

        factor_y = Factor(sample_data_with_nan, name="Y_with_nan")
        factor_x = Factor(data_x, name="X")

        neutralized = factor_y.cs_neutralize(factor_x)
        result_data = neutralized.data.to_pandas()

        # All values should be NaN due to strict NaN propagation
        assert result_data["factor"].isna().all()


class TestCSMean:
    """Test cs_mean (mean) operation with Polars support."""

    def test_cs_mean_basic_behavior(self, sample_data_minimal):
        """Test basic cs_mean computation on clean data."""
        factor_pd = Factor(sample_data_minimal, name="test_cs_mean")
        means = factor_pd.mean()

        # Values: 1, 2, 3. Mean=2. All should be 2.0
        expected = np.array([2.0, 2.0, 2.0])
        result_values = means.data.to_pandas()["factor"].values

        np.testing.assert_allclose(result_values, expected, rtol=1e-9, atol=1e-12)

    def test_cs_mean_nan_propagation(self, sample_data_with_nan):
        """Test that cs_mean returns all NaN when any input is NaN (strict mode)."""
        factor_pd = Factor(sample_data_with_nan, name="test_cs_mean_nan")
        means = factor_pd.mean()

        # All values should be NaN due to strict NaN propagation
        result_data = means.data.to_pandas()
        assert result_data["factor"].isna().all()


class TestCSMedian:
    """Test cs_median (median) operation with Polars support."""

    def test_cs_median_basic_behavior(self, sample_data_minimal):
        """Test basic cs_median computation on clean data."""
        factor_pd = Factor(sample_data_minimal, name="test_cs_median")
        medians = factor_pd.median()

        # Values: 1, 2, 3. Median=2. All should be 2.0
        expected = np.array([2.0, 2.0, 2.0])
        result_values = medians.data.to_pandas()["factor"].values

        np.testing.assert_allclose(result_values, expected, rtol=1e-9, atol=1e-12)

    def test_cs_median_nan_propagation(self, sample_data_with_nan):
        """Test that cs_median returns all NaN when any input is NaN (strict mode)."""
        factor_pd = Factor(sample_data_with_nan, name="test_cs_median_nan")
        medians = factor_pd.median()

        # All values should be NaN due to strict NaN propagation
        result_data = medians.data.to_pandas()
        assert result_data["factor"].isna().all()


class TestCSOpsPandasConsistency:
    """Test that Polars and Pandas implementations are numerically consistent."""

    def test_cs_rank_pandas_consistency(self, sample_data_multi_time):
        """Ensure cs_rank results match between Pandas and Polars."""
        factor = Factor(sample_data_multi_time, name="consistency_test")
        ranked = factor.cs_rank()

        # Verify the computation happened
        result = ranked.data.to_pandas()
        assert not result["factor"].isna().all()

    def test_cs_zscore_pandas_consistency(self, sample_data_multi_time):
        """Ensure cs_zscore results match between Pandas and Polars."""
        factor = Factor(sample_data_multi_time, name="consistency_test")
        zscored = factor.cs_zscore()

        # Verify the computation happened
        result = zscored.data.to_pandas()
        assert not result["factor"].isna().all()

    def test_cs_demean_pandas_consistency(self, sample_data_multi_time):
        """Ensure cs_demean results match between Pandas and Polars."""
        factor = Factor(sample_data_multi_time, name="consistency_test")
        demeaned = factor.cs_demean()

        # Verify the computation happened
        result = demeaned.data.to_pandas()
        assert not result["factor"].isna().all()

    def test_cs_winsorize_pandas_consistency(self):
        """Ensure cs_winsorize results match between Pandas and Polars."""
        data = pd.DataFrame(
            [
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "A", "factor": 1.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "B", "factor": 2.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "C", "factor": 3.0},
                {"start_time": 1609459200000, "end_time": 1609459200000, "symbol": "D", "factor": 100.0},
            ]
        )
        factor = Factor(data, name="consistency_test")
        winsorized = factor.cs_winsorize(limits=0.25)

        # Verify the computation happened
        result = winsorized.data.to_pandas()
        assert not result["factor"].isna().all()

    def test_cs_mean_pandas_consistency(self, sample_data_multi_time):
        """Ensure mean results match between Pandas and Polars."""
        factor = Factor(sample_data_multi_time, name="consistency_test")
        means = factor.mean()

        # Verify the computation happened
        result = means.data.to_pandas()
        assert not result["factor"].isna().all()

    def test_cs_median_pandas_consistency(self, sample_data_multi_time):
        """Ensure median results match between Pandas and Polars."""
        factor = Factor(sample_data_multi_time, name="consistency_test")
        medians = factor.median()

        # Verify the computation happened
        result = medians.data.to_pandas()
        assert not result["factor"].isna().all()


class TestCSOpLazyEvaluation:
    """Test that CS operations do not trigger LazyFrame.collect() during operation result building.

    These tests verify that the implementation maintains lazy evaluation semantics, only
    triggering .collect() when explicitly needed (e.g., when accessing .data property).

    These tests are currently expected to FAIL (RED) until production code is refactored
    to preserve LazyFrame semantics in cs_ops.
    """

    def test_cs_rank_does_not_collect_on_construction(self, sample_data_minimal, monkeypatch):
        """cs_rank should not trigger LazyFrame.collect() during result construction.

        The factor should remain lazy until .data is explicitly accessed.
        """
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        factor = Factor(sample_data_minimal, name="test_cs_rank_lazy")
        initial_count = collect_call_count["count"]

        # Building the result should NOT trigger collect
        result = factor.cs_rank()

        # Collect should not be called during operation (only during Factor.__init__)
        calls_during_op = collect_call_count["count"] - initial_count
        assert calls_during_op == 0, f"cs_rank triggered {calls_during_op} collect() calls, expected 0"

    def test_cs_winsorize_does_not_collect_on_construction(self, sample_data_minimal, monkeypatch):
        """cs_winsorize should not trigger LazyFrame.collect() during result construction.

        The factor should remain lazy until .data is explicitly accessed.
        """
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        factor = Factor(sample_data_minimal, name="test_cs_winsorize_lazy")
        initial_count = collect_call_count["count"]

        # Building the result should NOT trigger collect
        result = factor.cs_winsorize(limits=0.025)

        # Collect should not be called during operation (only during Factor.__init__)
        calls_during_op = collect_call_count["count"] - initial_count
        assert calls_during_op == 0, f"cs_winsorize triggered {calls_during_op} collect() calls, expected 0"

    def test_cs_neutralize_does_not_collect_on_construction(self, sample_data_minimal, monkeypatch):
        """cs_neutralize should not trigger LazyFrame.collect() during result construction.

        The factor should remain lazy until .data is explicitly accessed.
        """
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        # Create two factors
        factor_y = Factor(sample_data_minimal, name="Y_lazy")
        factor_x = Factor(sample_data_minimal, name="X_lazy")
        initial_count = collect_call_count["count"]

        # Building the result should NOT trigger collect
        result = factor_y.cs_neutralize(factor_x)

        # Collect should not be called during operation (only during Factor.__init__)
        calls_during_op = collect_call_count["count"] - initial_count
        assert calls_during_op == 0, f"cs_neutralize triggered {calls_during_op} collect() calls, expected 0"

    def test_cs_zscore_does_not_collect_on_construction(self, sample_data_minimal, monkeypatch):
        """cs_zscore should not trigger LazyFrame.collect() during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        factor = Factor(sample_data_minimal, name="test_cs_zscore_lazy")
        initial_count = collect_call_count["count"]

        # Building the result should NOT trigger collect
        result = factor.cs_zscore()

        # Collect should not be called during operation (only during Factor.__init__)
        calls_during_op = collect_call_count["count"] - initial_count
        assert calls_during_op == 0, f"cs_zscore triggered {calls_during_op} collect() calls, expected 0"

    def test_cs_demean_does_not_collect_on_construction(self, sample_data_minimal, monkeypatch):
        """cs_demean should not trigger LazyFrame.collect() during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        factor = Factor(sample_data_minimal, name="test_cs_demean_lazy")
        initial_count = collect_call_count["count"]

        # Building the result should NOT trigger collect
        result = factor.cs_demean()

        # Collect should not be called during operation (only during Factor.__init__)
        calls_during_op = collect_call_count["count"] - initial_count
        assert calls_during_op == 0, f"cs_demean triggered {calls_during_op} collect() calls, expected 0"

    def test_cs_mean_does_not_collect_on_construction(self, sample_data_minimal, monkeypatch):
        """cs_mean should not trigger LazyFrame.collect() during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        factor = Factor(sample_data_minimal, name="test_cs_mean_lazy")
        initial_count = collect_call_count["count"]

        # Building the result should NOT trigger collect
        result = factor.mean()

        # Collect should not be called during operation (only during Factor.__init__)
        calls_during_op = collect_call_count["count"] - initial_count
        assert calls_during_op == 0, f"mean triggered {calls_during_op} collect() calls, expected 0"

    def test_cs_median_does_not_collect_on_construction(self, sample_data_minimal, monkeypatch):
        """cs_median should not trigger LazyFrame.collect() during result construction."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        factor = Factor(sample_data_minimal, name="test_cs_median_lazy")
        initial_count = collect_call_count["count"]

        # Building the result should NOT trigger collect
        result = factor.median()

        # Collect should not be called during operation (only during Factor.__init__)
        calls_during_op = collect_call_count["count"] - initial_count
        assert calls_during_op == 0, f"median triggered {calls_during_op} collect() calls, expected 0"

    def test_cs_operations_chain_multiple_lazy_operations(self, sample_data_minimal, monkeypatch):
        """Chained CS operations should accumulate lazy operations without collecting."""
        collect_call_count = {"count": 0}
        original_collect = pl.LazyFrame.collect

        def tracked_collect(self, *args, **kwargs):
            collect_call_count["count"] += 1
            return original_collect(self, *args, **kwargs)

        monkeypatch.setattr(pl.LazyFrame, "collect", tracked_collect)

        factor = Factor(sample_data_minimal, name="test_chaining_lazy")
        initial_count = collect_call_count["count"]

        # Chain multiple operations - should not collect until .data access
        result = factor.cs_rank().cs_demean()

        # Collect should not be called during operations (only during Factor.__init__)
        calls_during_ops = collect_call_count["count"] - initial_count
        assert calls_during_ops == 0, f"Chained operations triggered {calls_during_ops} collect() calls, expected 0"

        # Accessing .data should trigger collect
        _ = result.data
        assert collect_call_count["count"] > initial_count, "collect() should be called when accessing .data"
