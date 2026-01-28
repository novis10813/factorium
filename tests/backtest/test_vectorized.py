"""Tests for VectorizedBacktester."""

import pytest
import polars as pl
import numpy as np

from factorium import AggBar
from factorium.backtest.vectorized import VectorizedBacktester, BacktestResult


class TestVectorizedBacktesterInit:
    """Tests for VectorizedBacktester initialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 50, 3600000))

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": price * 0.99,
                        "high": price * 1.01,
                        "low": price * 0.98,
                        "close": price,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_init_with_aggbar(self, sample_data):
        """Should initialize with AggBar."""
        signal = sample_data["close"].cs_rank()

        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
        )

        assert bt.initial_capital == 10000.0
        assert bt.neutralization == "market"

    def test_run_returns_result(self, sample_data):
        """run() should return BacktestResult."""
        signal = sample_data["close"].cs_rank()

        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert result.returns is not None
        assert result.metrics is not None

    def test_equity_curve_is_polars_dataframe(self, sample_data):
        """equity_curve should be Polars DataFrame."""
        signal = sample_data["close"].cs_rank()

        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert isinstance(result.equity_curve, pl.DataFrame)
        assert "end_time" in result.equity_curve.columns
        assert "total_value" in result.equity_curve.columns

    def test_total_value_positive(self, sample_data):
        """Total value should always be positive."""
        signal = sample_data["close"].cs_rank()

        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert result.equity_curve["total_value"].min() > 0


class TestWeightCalculation:
    """Tests for weight calculation in VectorizedBacktester."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known signals."""
        timestamps = [1704067200000, 1704070800000, 1704074400000]

        rows = []
        for ts in timestamps:
            # Create signals that sum to known values
            for symbol, signal in [("A", 0.8), ("B", 0.5), ("C", 0.2), ("D", -0.1)]:
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": 100.0,
                        "high": 100.0,
                        "low": 100.0,
                        "close": 100.0,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_market_neutral_weights_sum_to_zero(self, sample_data):
        """Market neutral weights should sum to zero."""
        signal = sample_data["close"]  # Will be constant, but test the logic

        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
            neutralization="market",
        )

        # Access internal method
        combined = bt._prepare_data()
        weighted = bt._calculate_weights(combined)

        # Group by end_time and check sum
        weight_sums = weighted.group_by("end_time").agg(pl.col("weight").sum().alias("weight_sum"))

        # All sums should be approximately zero
        assert weight_sums["weight_sum"].abs().max() < 1e-10

    def test_long_only_weights_sum_to_one(self):
        """Long-only weights should sum to 1."""
        timestamps = [1704067200000, 1704070800000]

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol, price in [("A", 100.0), ("B", 50.0), ("C", 25.0)]:
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "close": price * (1 + 0.01 * i),  # Varying prices
                        "open": price,
                        "high": price,
                        "low": price,
                        "volume": 1000.0,
                    }
                )

        data = AggBar(pl.DataFrame(rows))
        signal = data["close"].cs_rank()

        bt = VectorizedBacktester(
            prices=data,
            signal=signal,
            neutralization="none",
        )

        combined = bt._prepare_data()
        weighted = bt._calculate_weights(combined)

        # Group by end_time and check sum (excluding first row which has no prev_signal)
        weight_sums = (
            weighted.filter(pl.col("weight") != 0).group_by("end_time").agg(pl.col("weight").sum().alias("weight_sum"))
        )

        # All non-zero sums should be approximately 1
        for ws in weight_sums["weight_sum"].to_list():
            if ws > 0:
                assert abs(ws - 1.0) < 1e-10


class TestMetricsCalculation:
    """Tests for metrics calculation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 50, 3600000))

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": price * 0.99,
                        "high": price * 1.01,
                        "low": price * 0.98,
                        "close": price,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_metrics_include_sortino(self, sample_data):
        """Metrics should include Sortino ratio."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert "sortino_ratio" in result.metrics

    def test_metrics_include_calmar(self, sample_data):
        """Metrics should include Calmar ratio."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert "calmar_ratio" in result.metrics

    def test_metrics_include_win_rate(self, sample_data):
        """Metrics should include win rate."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert "win_rate" in result.metrics

    def test_max_drawdown_non_positive(self, sample_data):
        """Max drawdown should be <= 0."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert result.metrics["max_drawdown"] <= 0


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing Backtester API."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 20, 3600000))

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH"]:
                base_price = {"BTC": 100.0, "ETH": 50.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": price * 0.99,
                        "high": price * 1.01,
                        "low": price * 0.98,
                        "close": price,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_can_import_from_backtest_module(self):
        """Should be able to import VectorizedBacktester from backtest."""
        from factorium.backtest import VectorizedBacktester

        assert VectorizedBacktester is not None

    def test_result_to_pandas_compatibility(self, sample_data):
        """BacktestResult.to_pandas() should return pandas DataFrames."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        pandas_result = result.to_pandas()

        import pandas as pd

        assert isinstance(pandas_result.equity_curve, pd.DataFrame)
        assert isinstance(pandas_result.returns, pd.DataFrame)
        assert isinstance(pandas_result.metrics, dict)


class TestConstraintIntegration:
    """Tests for constraint integration in VectorizedBacktester."""

    @pytest.fixture
    def sample_data(self):
        from factorium import AggBar
        import polars as pl

        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 20, 3600000))

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "close": price,
                        "open": price,
                        "high": price,
                        "low": price,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_max_position_constraint(self, sample_data):
        """Should apply MaxPositionConstraint."""
        from factorium.backtest import MaxPositionConstraint, VectorizedBacktester

        signal = sample_data["close"].cs_rank()
        constraint = MaxPositionConstraint(max_weight=0.1)

        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
            constraints=[constraint],
        )
        # Verify constraints is stored
        assert bt.constraints == [constraint]

        result = bt.run()
        assert result.metrics is not None

    def test_long_only_constraint(self, sample_data):
        """Should apply LongOnlyConstraint."""
        from factorium.backtest import LongOnlyConstraint, VectorizedBacktester

        signal = sample_data["close"].cs_rank()
        constraint = LongOnlyConstraint()

        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
            neutralization="market",  # This creates negative weights
            constraints=[constraint],
        )
        result = bt.run()

        assert result.metrics is not None
