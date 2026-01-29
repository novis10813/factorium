import pytest
import tempfile
from pathlib import Path
from factorium import AggBar
from factorium.factors import Factor
from factorium.factors.analyzer import FactorAnalysisResult


@pytest.fixture
def sample_factor_and_prices():
    """Fixture providing sample factor and price data for testing."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT"]

    data = []
    for date in dates:
        for symbol in symbols:
            data.append(
                {
                    "start_time": int(date.timestamp() * 1000),
                    "end_time": int((date + pd.Timedelta(days=1)).timestamp() * 1000),
                    "symbol": symbol,
                    "close": np.random.randn() * 10 + 100,  # Random prices around 100
                    "my_factor": np.random.randn(),  # Random factor values
                }
            )

    df = pd.DataFrame(data)
    agg = AggBar(df)
    factor = agg["my_factor"]
    prices = agg["close"]

    return factor, prices


def test_factor_eval_returns_analysis_result(sample_factor_and_prices):
    """Test that Factor.eval() returns FactorAnalysisResult."""
    factor, prices = sample_factor_and_prices

    result = factor.eval(prices, periods=1, quantiles=5)

    assert isinstance(result, FactorAnalysisResult)
    assert result.factor_name == factor.name
    assert result.periods == 1
    assert result.quantiles == 5
    assert hasattr(result, "turnover_series")
    assert hasattr(result, "turnover_mean")


def test_factor_eval_with_output_dir(sample_factor_and_prices):
    """Test that Factor.eval() creates output when output_dir is specified."""
    factor, prices = sample_factor_and_prices

    with tempfile.TemporaryDirectory() as tmpdir:
        result = factor.eval(prices, periods=1, output_dir=tmpdir)

        # Check that experiment folder was created
        exp_dirs = list(Path(tmpdir).glob("*_*"))
        assert len(exp_dirs) == 1

        # Check config.json exists
        config_path = exp_dirs[0] / "config.json"
        assert config_path.exists()
