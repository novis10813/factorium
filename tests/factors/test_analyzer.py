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
    p1_returns = df["period_1"].dropna()
    assert not p1_returns.empty


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
