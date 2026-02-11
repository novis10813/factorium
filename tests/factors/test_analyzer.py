import pandas as pd
import numpy as np
import pytest
from factorium import AggBar
from factorium.factors import Factor
from factorium.factors.analyzer import FactorAnalyzer


@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    symbols = ["AAPL", "GOOGL"]

    data = []
    for date in dates:
        for symbol in symbols:
            data.append(
                {
                    "start_time": int(date.timestamp() * 1000),
                    "end_time": int((date + pd.Timedelta(days=1)).timestamp() * 1000),
                    "symbol": symbol,
                    "close": np.random.randn() + 100,
                    "my_factor": np.random.randn(),
                }
            )
    return pd.DataFrame(data)


def test_analyzer_initialization(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    assert analyzer.factor == factor
    assert isinstance(analyzer.prices, Factor)


def test_prepare_data(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg  # Test AggBar to Factor conversion

    analyzer = FactorAnalyzer(factor, prices)
    periods = [1, 2]
    df = analyzer.prepare_data(periods=periods, price_col="close")

    assert "factor" in df.columns
    for p in periods:
        assert f"period_{p}" in df.columns

    # Check if returns are calculated correctly for period 1
    # Return = (prices.shift(-1) - prices) / prices
    if isinstance(df, pd.DataFrame):
        p1_returns = df["period_1"].dropna()
        assert not p1_returns.empty
    else:
        # Polars
        p1_returns = df["period_1"].drop_nulls()
        assert len(p1_returns) > 0


def test_prepare_data_empty_factor():
    # Create an empty factor
    empty_df = pd.DataFrame(columns=["start_time", "end_time", "symbol", "factor"])
    factor = Factor(empty_df)

    # Create some price data
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    price_data = pd.DataFrame(
        {
            "start_time": [int(d.timestamp() * 1000) for d in dates],
            "end_time": [int((d + pd.Timedelta(days=1)).timestamp() * 1000) for d in dates],
            "symbol": "AAPL",
            "close": [100.0] * 5,
        }
    )
    prices = Factor(price_data)

    analyzer = FactorAnalyzer(factor, prices)
    with pytest.raises(ValueError, match="Factor data is empty."):
        analyzer.prepare_data()


def test_calculate_ic(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    periods = [1, 2]
    analyzer.prepare_data(periods=periods)

    ic = analyzer.calculate_ic(method="rank")

    assert isinstance(ic, pd.DataFrame)
    for p in periods:
        col = f"period_{p}"
        assert col in ic.columns
        # IC should be between -1 and 1
        assert ic[col].min() >= -1.0
        assert ic[col].max() <= 1.0


def test_ic_summary(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    analyzer.prepare_data(periods=[1])

    summary = analyzer.calculate_ic_summary()
    assert isinstance(summary, pd.DataFrame)
    assert "period_1" in summary.columns
    assert "mean" in summary.index
    assert "std" in summary.index
    assert "t-stat" in summary.index
    assert "ic_ir" in summary.index


def test_calculate_quantile_returns(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    analyzer.prepare_data(periods=[1])

    q_ret = analyzer.calculate_quantile_returns(quantiles=2, period=1)

    assert isinstance(q_ret, pd.DataFrame)
    assert "mean_ret" in q_ret.columns
    assert "count" in q_ret.columns
    # Check if we have 2 quantiles (if data allows)
    assert q_ret.index.get_level_values("quantile").nunique() <= 2


def test_calculate_cumulative_returns(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    analyzer.prepare_data(periods=[1])

    cum_ret = analyzer.calculate_cumulative_returns(quantiles=2, period=1, long_short=True)

    assert isinstance(cum_ret, pd.DataFrame)
    if not cum_ret.empty:
        # Check for quantile columns and Long-Short
        assert "Long-Short" in cum_ret.columns
        assert 1 in cum_ret.columns
        assert 2 in cum_ret.columns


def test_plotting(sample_data):
    import matplotlib.figure as mpl_figure

    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    analyzer.prepare_data(periods=[1, 5])

    # Test plot_ic time series
    fig_ic_ts = analyzer.plot_ic(period=1, plot_type="ts")
    assert isinstance(fig_ic_ts, mpl_figure.Figure)

    # Test plot_ic histogram
    fig_ic_hist = analyzer.plot_ic(period=1, plot_type="hist")
    assert isinstance(fig_ic_hist, mpl_figure.Figure)

    # Test invalid plot_type
    with pytest.raises(ValueError, match="Invalid plot_type"):
        analyzer.plot_ic(period=1, plot_type="invalid")

    # Test plot_quantile_returns
    fig_q = analyzer.plot_quantile_returns(quantiles=2, period=1)
    assert isinstance(fig_q, mpl_figure.Figure)

    # Test plot_cumulative_returns
    fig_cum = analyzer.plot_cumulative_returns(quantiles=2, period=1)
    assert isinstance(fig_cum, mpl_figure.Figure)


def test_analyze_returns_dataclass(sample_data):
    """analyze() should return FactorAnalysisResult dataclass."""
    from factorium.factors.analyzer import FactorAnalysisResult

    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    result = analyzer.analyze(periods=1)

    assert isinstance(result, FactorAnalysisResult)
    assert result.factor_name == "my_factor"
    assert result.periods == 1
    assert "mean_ic" in result.ic_summary
    assert hasattr(result, "to_dict")
