import numpy as np
import pandas as pd
import polars as pl
import pytest

from factorium import AggBar, Factor
from factorium.backtest import (
    Backtester,
    BacktestResult,
    LegacyBacktester,
    LegacyBacktestResult,
    Portfolio,
    VectorizedBacktester,
    calculate_metrics,
    frequency_to_periods_per_year,
    neutralize_weights,
    normalize_weights,
    parse_frequency_to_seconds,
)
from factorium.backtest.allocators import LongOnlyAllocator, MarketNeutralAllocator
from factorium.backtest.pipeline import AlphaPipeline


class TestNeutralizeWeights:
    def test_basic_neutralization(self):
        signals = pd.Series([0.8, 0.5, 0.3, 0.1], index=["A", "B", "C", "D"])
        weights = neutralize_weights(signals)

        assert abs(weights.sum()) < 1e-10
        assert abs(weights.abs().sum() - 1.0) < 1e-10

    def test_empty_signals(self):
        signals = pd.Series(dtype=float)
        weights = neutralize_weights(signals)
        assert len(weights) == 0

    def test_all_nan(self):
        signals = pd.Series([np.nan, np.nan, np.nan])
        weights = neutralize_weights(signals)
        assert len(weights) == 0

    def test_partial_nan(self):
        signals = pd.Series([1.0, np.nan, 3.0], index=["A", "B", "C"])
        weights = neutralize_weights(signals)
        assert "B" not in weights.index
        assert abs(weights.sum()) < 1e-10


class TestNormalizeWeights:
    def test_basic_normalization(self):
        signals = pd.Series([2.0, 3.0, 5.0], index=["A", "B", "C"])
        weights = normalize_weights(signals)

        assert abs(weights.sum() - 1.0) < 1e-10

    def test_filters_negative_values(self):
        signals = pd.Series([2.0, -3.0, 5.0], index=["A", "B", "C"])
        weights = normalize_weights(signals)

        assert "B" not in weights.index
        assert abs(weights.sum() - 1.0) < 1e-10
        assert len(weights) == 2


class TestFrequencyParsing:
    def test_parse_seconds(self):
        assert parse_frequency_to_seconds("30s") == 30

    def test_parse_minutes(self):
        assert parse_frequency_to_seconds("10m") == 600

    def test_parse_hours(self):
        assert parse_frequency_to_seconds("1h") == 3600

    def test_parse_days(self):
        assert parse_frequency_to_seconds("1d") == 86400

    def test_periods_per_year_hourly(self):
        ppy = frequency_to_periods_per_year("1h")
        assert abs(ppy - 365.25 * 24) < 1

    def test_periods_per_year_daily(self):
        ppy = frequency_to_periods_per_year("1d")
        assert abs(ppy - 365.25) < 0.01

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid frequency"):
            parse_frequency_to_seconds("invalid")


class TestCalculateMetrics:
    def test_positive_returns(self):
        returns = pd.Series([0.01, 0.02, 0.01, -0.005, 0.015] * 100)
        metrics = calculate_metrics(returns)

        assert metrics["total_return"] > 0
        assert metrics["sharpe_ratio"] > 0
        assert metrics["max_drawdown"] <= 0

    def test_empty_returns(self):
        returns = pd.Series(dtype=float)
        metrics = calculate_metrics(returns)
        assert np.isnan(metrics["total_return"])

    def test_all_nan_returns(self):
        returns = pd.Series([np.nan, np.nan, np.nan])
        metrics = calculate_metrics(returns)
        assert np.isnan(metrics["total_return"])


class TestPortfolio:
    def test_initial_state(self):
        portfolio = Portfolio(initial_capital=10000.0)
        assert portfolio.cash == 10000.0
        assert len(portfolio.positions) == 0

    def test_buy_trade(self):
        portfolio = Portfolio(initial_capital=10000.0)
        portfolio.execute_trade("BTC", 1.0, 100.0, (0.001, 0.001), 1000)

        assert portfolio.cash < 10000.0
        assert portfolio.positions["BTC"] == 1.0
        assert len(portfolio.trade_log) == 1

    def test_sell_trade(self):
        portfolio = Portfolio(initial_capital=10000.0)
        portfolio.execute_trade("BTC", 1.0, 100.0, (0.001, 0.001), 1000)
        portfolio.execute_trade("BTC", -1.0, 110.0, (0.001, 0.001), 2000)

        assert "BTC" not in portfolio.positions
        assert len(portfolio.trade_log) == 2

    def test_market_value(self):
        portfolio = Portfolio(initial_capital=10000.0)
        portfolio.execute_trade("BTC", 2.0, 100.0, (0.0, 0.0), 1000)

        prices = pd.Series({"BTC": 150.0})
        assert portfolio.get_market_value(prices) == 300.0


class TestBacktester:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2025-01-01", periods=20, freq="1h")
        timestamps = dates.astype("datetime64[ms]").astype(np.int64)

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH"]:
                base_price = 100.0 if symbol == "BTC" else 50.0
                price = base_price * (1 + 0.01 * i + 0.005 * np.random.randn())
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

        df = pl.DataFrame(rows)
        return AggBar(df)

    def test_basic_backtest(self, sample_data):
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = Backtester(
            prices=sample_data,
            signal=signal,
            transaction_cost=0.0001,
            initial_capital=10000.0,
            pipeline=AlphaPipeline(allocator=MarketNeutralAllocator()),
        )

        result = bt.run()

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert len(result.returns) > 0
        assert "sharpe_ratio" in result.metrics

    def test_summary(self, sample_data):
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = Backtester(prices=sample_data, signal=signal)
        bt.run()

        summary = bt.summary()

        assert "initial_capital" in summary
        assert "final_value" in summary
        assert "total_turnover" in summary
        assert "total_cost" in summary
        assert "sharpe_ratio" in summary
        # num_trades is removed in return-based mode
        assert "num_trades" not in summary

    def test_no_lookahead_bias(self, sample_data):
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = Backtester(prices=sample_data, signal=signal)
        result = bt.run()

        # Weights should exist and have data
        assert len(result.weights) > 0
        # First timestamp should have zero weights (no prev signal)
        first_time = result.weights["end_time"].min()
        first_weights = result.weights.filter(pl.col("end_time") == first_time)["weight"]
        assert first_weights.abs().sum() < 1e-10, "First period should have zero weights (no previous signal)"

    def test_invalid_entry_price(self, sample_data):
        signal = sample_data["close"].cs_rank()

        with pytest.raises(ValueError, match="entry_price"):
            # Should raise during initialization
            Backtester(prices=sample_data, signal=signal, entry_price="invalid")

    def test_cost_rates_tuple(self, sample_data):
        signal = sample_data["close"].cs_rank()

        bt = Backtester(
            prices=sample_data,
            signal=signal,
            transaction_cost=(0.0003, 0.0005),
        )
        result = bt.run()

        assert isinstance(result, BacktestResult)

    def test_frequency_parameter(self, sample_data):
        signal = sample_data["close"].cs_rank()

        bt_hourly = Backtester(prices=sample_data, signal=signal, frequency="1h")
        bt_daily = Backtester(prices=sample_data, signal=signal, frequency="1d")

        bt_hourly.run()
        bt_daily.run()

        assert bt_hourly._periods_per_year > bt_daily._periods_per_year


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_single_symbol_backtest(self):
        """Single asset should work without cross-sectional operations failing."""
        dates = pd.date_range(start="2025-01-01", periods=20, freq="1h")
        timestamps = dates.astype("datetime64[ms]").astype(np.int64)

        rows = []
        for i, ts in enumerate(timestamps):
            price = 100.0 * (1 + 0.01 * i)
            rows.append(
                {
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": "BTC",
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1000.0,
                }
            )

        df = pd.DataFrame(rows)
        agg = AggBar(df)
        signal = agg["close"].cs_rank()

        bt = Backtester(prices=agg, signal=signal, pipeline=AlphaPipeline(allocator=LongOnlyAllocator()))
        result = bt.run()

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_identical_signals_weights(self):
        """All identical signals should produce equal weights."""
        weights = normalize_weights(pd.Series([1.0, 1.0, 1.0], index=["A", "B", "C"]))
        assert abs(weights["A"] - weights["B"]) < 1e-10
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_neutralize_weights_with_identical_signals(self):
        """Identical signals after neutralization should be zero (market neutral)."""
        signals = pd.Series([1.0, 1.0, 1.0, 1.0], index=["A", "B", "C", "D"])
        weights = neutralize_weights(signals)

        assert abs(weights.sum()) < 1e-10
        assert all(abs(w) < 1e-10 for w in weights)

    def test_periods_per_year_validation(self):
        """Invalid periods_per_year should raise ValueError."""
        returns = pd.Series([0.01, 0.02, -0.01])

        with pytest.raises(ValueError, match="periods_per_year"):
            calculate_metrics(returns, periods_per_year=0.5)

        with pytest.raises(ValueError, match="periods_per_year"):
            calculate_metrics(returns, periods_per_year=1e10)


class TestVectorizedBacktesterIntegration:
    """Integration tests comparing VectorizedBacktester with Backtester."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", periods=20, freq="1h")
        timestamps = dates.astype("datetime64[ms]").astype(np.int64)

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH"]:
                base_price = 100.0 if symbol == "BTC" else 50.0
                price = base_price * (1 + 0.01 * i + 0.005 * np.random.randn())
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

        df = pl.DataFrame(rows)
        return AggBar(df)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_vectorized_produces_reasonable_equity(self, sample_data):
        """VectorizedBacktester should produce a reasonable equity curve."""
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
            transaction_cost=0.0001,
            initial_capital=10000.0,
            pipeline=AlphaPipeline(allocator=MarketNeutralAllocator()),
        )
        result = bt.run()

        # Equity should start near initial capital
        first_value = result.equity_curve["total_value"][0]
        assert abs(first_value - 10000.0) < 100.0  # within 1% on first period

        # Should have returns for each period
        assert len(result.returns) == len(result.equity_curve)

    def test_vectorized_polars_output_types(self, sample_data):
        """VectorizedBacktester should return Polars DataFrames with correct fields."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert isinstance(result.equity_curve, pl.DataFrame)
        assert isinstance(result.returns, pl.DataFrame)
        assert isinstance(result.weights, pl.DataFrame)
        assert isinstance(result.turnover, pl.DataFrame)

        # Check column names
        assert set(result.equity_curve.columns) == {"end_time", "total_value"}
        assert set(result.returns.columns) == {"end_time", "return"}
        assert set(result.weights.columns) == {"end_time", "symbol", "weight"}
        assert set(result.turnover.columns) == {"end_time", "turnover", "cost"}

        # Should NOT have trades or portfolio_history
        assert not hasattr(result, "trades")
        assert not hasattr(result, "portfolio_history")

    def test_backtest_result_pandas_importable(self):
        """BacktestResultPandas should be importable from factorium.backtest."""
        from factorium.backtest import BacktestResultPandas

        assert BacktestResultPandas is not None

    def test_vectorized_metrics_complete(self, sample_data):
        """Metrics should contain all expected keys."""
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        expected_keys = {
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "max_drawdown",
            "var_95",
            "cvar_95",
            "win_rate",
            "profit_factor",
        }
        assert expected_keys.issubset(result.metrics.keys())


class TestMissingPriceHandling:
    """Tests for handling missing prices."""

    def test_missing_price_symbol_excluded_from_holdings(self):
        """Symbols with missing prices should be excluded from target holdings."""
        dates = pd.date_range(start="2025-01-01", periods=10, freq="1h")
        timestamps = dates.astype("datetime64[ms]").astype(np.int64)

        rows = []
        for i, ts in enumerate(timestamps):
            # BTC has all prices
            rows.append(
                {
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": "BTC",
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "volume": 1000.0,
                }
            )
            # ETH only has prices for first 5 bars
            if i < 5:
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": "ETH",
                        "open": 50.0,
                        "high": 50.0,
                        "low": 50.0,
                        "close": 50.0,
                        "volume": 1000.0,
                    }
                )

        df = pl.DataFrame(rows)
        agg = AggBar(df)

        # Signal includes both symbols
        signal = agg["close"].cs_rank()

        bt = Backtester(
            prices=agg,
            signal=signal,
            pipeline=AlphaPipeline(allocator=MarketNeutralAllocator()),
        )
        result = bt.run()

        # After bar 5, ETH should have zero weight (no price data)
        eth_weights_after_5 = result.weights.filter(
            (pl.col("symbol") == "ETH") & (pl.col("end_time") > timestamps[4] + 3600000)
        )
        # ETH weights should be 0 or absent when price data is missing
        if len(eth_weights_after_5) > 0:
            assert eth_weights_after_5["weight"].abs().sum() < 1e-10


class TestLegacyBacktester:
    """Tests for the legacy iterative backtester."""

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2025-01-01", periods=20, freq="1h")
        timestamps = dates.astype("datetime64[ms]").astype(np.int64)

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH"]:
                base_price = 100.0 if symbol == "BTC" else 50.0
                price = base_price * (1 + 0.01 * i + 0.005 * np.random.randn())
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

        df = pd.DataFrame(rows)
        return AggBar(df)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_legacy_basic_backtest(self, sample_data):
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = LegacyBacktester(
            prices=sample_data,
            signal=signal,
            transaction_cost=0.0001,
            initial_capital=10000.0,
            neutralization="market",
        )

        result = bt.run()

        assert isinstance(result, LegacyBacktestResult)
        assert len(result.equity_curve) > 0
        assert len(result.returns) > 0
        assert "sharpe_ratio" in result.metrics
