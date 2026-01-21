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
