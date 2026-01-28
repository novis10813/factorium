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
