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
        timestamps = dates.astype(np.int64) // 10**6

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
            neutralization="market",
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
        assert "num_trades" in summary
        assert "sharpe_ratio" in summary

    def test_no_lookahead_bias(self, sample_data):
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = Backtester(prices=sample_data, signal=signal)
        result = bt.run()

        assert len(result.trades) > 0
        # Vectorized result uses end_time instead of timestamp, and is Polars
        first_trade_ts = result.trades["end_time"][0]
        first_signal_ts = signal.data["end_time"].min()
        assert first_trade_ts > first_signal_ts

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
        timestamps = dates.astype(np.int64) // 10**6

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

        bt = Backtester(prices=agg, signal=signal, neutralization="none")
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
        timestamps = dates.astype(np.int64) // 10**6

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

    def test_vectorized_vs_original_equity_curve(self, sample_data):
        """VectorizedBacktester should produce similar equity curve to Backtester."""
        close = sample_data["close"]
        signal = close.cs_rank()

        # Run original backtester
        bt_orig = LegacyBacktester(
            prices=sample_data,
            signal=signal,
            transaction_cost=0.0001,
            initial_capital=10000.0,
            neutralization="market",
        )
        result_orig = bt_orig.run()

        # Run vectorized backtester
        bt_vec = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
            transaction_cost=0.0001,
            initial_capital=10000.0,
            neutralization="market",
        )
        result_vec = bt_vec.run()

        # Compare final equity
        final_orig = result_orig.equity_curve.iloc[-1]

        # Vectorized result is Polars
        final_vec = result_vec.equity_curve["total_value"].to_list()[-1]

        # Use 1% tolerance
        assert abs(final_vec - final_orig) / final_orig < 0.01

    def test_vectorized_polars_output_types(self, sample_data):
        """VectorizedBacktester should return Polars DataFrames."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        import polars as pl

        assert isinstance(result.equity_curve, pl.DataFrame)
        assert isinstance(result.returns, pl.DataFrame)
        assert isinstance(result.trades, pl.DataFrame)

    def test_vectorized_metrics_comparable(self, sample_data):
        """Metrics should be comparable between implementations."""
        close = sample_data["close"]
        signal = close.cs_rank()

        bt_orig = Backtester(prices=sample_data, signal=signal)
        result_orig = bt_orig.run()

        bt_vec = VectorizedBacktester(prices=sample_data, signal=signal)
        result_vec = bt_vec.run()

        # Compare sharpe_ratio
        assert abs(result_vec.metrics["sharpe_ratio"] - result_orig.metrics["sharpe_ratio"]) < 0.1
        # Compare total_return
        assert abs(result_vec.metrics["total_return"] - result_orig.metrics["total_return"]) < 0.01


class TestBacktesterCashHandling:
    def test_cash_never_negative(self):
        # BTC starts cheap, becomes very expensive
        # ETH stays cheap
        dates = pd.date_range(start="2025-01-01", periods=10, freq="1h")
        timestamps = dates.astype(np.int64) // 10**6
        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH"]:
                if symbol == "BTC":
                    price = 10.0 if i < 5 else 10000.0  # Price jumps 1000x
                else:
                    price = 10.0
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 1000.0,
                    }
                )

        df = pl.DataFrame(rows)
        agg = AggBar(df)

        # Signal always wants to buy BTC
        signal = agg["close"].cs_rank()

        bt = Backtester(
            prices=agg,
            signal=signal,
            initial_capital=1000.0,  # Not enough for expensive BTC
            neutralization="none",
            transaction_cost=0.0,
        )
        result = bt.run()

        # Should complete without error
        assert isinstance(result, BacktestResult)
        # Cash should never go negative
        assert result.portfolio_history["cash"].min() >= -1e-10


class TestMissingPriceHandling:
    """Tests for handling missing prices."""

    def test_missing_price_symbol_excluded_from_holdings(self):
        """Symbols with missing prices should be excluded from target holdings."""
        dates = pd.date_range(start="2025-01-01", periods=10, freq="1h")
        timestamps = dates.astype(np.int64) // 10**6

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
            neutralization="market",
        )
        result = bt.run()

        # After bar 5, ETH should have no trades
        eth_trades_after_5 = result.trades.filter(
            (pl.col("symbol") == "ETH") & (pl.col("end_time") > timestamps[4] + 3600000)
        )
        assert len(eth_trades_after_5) == 0


class TestLegacyBacktester:
    """Tests for the legacy iterative backtester."""

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2025-01-01", periods=20, freq="1h")
        timestamps = dates.astype(np.int64) // 10**6

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
