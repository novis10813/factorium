import pandas as pd
import polars as pl
import numpy as np
import pytest
from unittest.mock import patch
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


def test_analyzer_uses_polars_internally(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)

    # Mock Factor.to_pandas to ensure it's not called
    with patch.object(Factor, "to_pandas", side_effect=RuntimeError("Factor.to_pandas() called!")) as mock_to_pandas:
        # Should not raise RuntimeError if implemented correctly
        df = analyzer.prepare_data(periods=[1])
        assert isinstance(df, pl.DataFrame)
        mock_to_pandas.assert_not_called()


def test_calculate_ic_uses_polars_internally(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    analyzer.prepare_data(periods=[1])

    # Mock Factor.to_pandas is not enough here since _clean_data is already Polars
    # We want to ensure it doesn't use Pandas for grouping/correlation
    with patch("pandas.DataFrame.groupby") as mock_groupby:
        ic = analyzer.calculate_ic()
        assert isinstance(ic, pd.DataFrame)
        # Should not use pandas groupby
        mock_groupby.assert_not_called()


def test_calculate_quantile_returns_uses_polars_internally(sample_data):
    agg = AggBar(sample_data)
    factor = agg["my_factor"]
    prices = agg["close"]

    analyzer = FactorAnalyzer(factor, prices)
    analyzer.prepare_data(periods=[1])

    with patch("pandas.DataFrame.groupby") as mock_groupby:
        q_ret = analyzer.calculate_quantile_returns(quantiles=2)
        assert isinstance(q_ret, pd.DataFrame)
        # Should not use pandas groupby
        mock_groupby.assert_not_called()
