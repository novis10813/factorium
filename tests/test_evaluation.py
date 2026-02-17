import pandas as pd
import numpy as np
import pytest
from factorium.factors.core import Factor
from factorium.factors.analyzer import FactorAnalysisResult


def test_factor_evaluation_flow():
    # Create dummy price data (upward trend)
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    symbols = ["AAPL", "MSFT", "GOOG"]

    price_data = []
    for d in dates:
        for s in symbols:
            # Price increases over time
            p = 100 + (d - dates[0]).days * 2 + np.random.randn()
            price_data.append([d, d, s, p])

    prices_df = pd.DataFrame(price_data, columns=["start_time", "end_time", "symbol", "factor"])
    prices_factor = Factor(prices_df, name="close")

    # Create dummy signal factor
    signal_data = []
    for i, d in enumerate(dates):
        for s in symbols:
            sig = np.random.randn()
            signal_data.append([d, d, s, sig])

    signal_df = pd.DataFrame(signal_data, columns=["start_time", "end_time", "symbol", "factor"])
    signal_factor = Factor(signal_df, name="signal")

    # Run eval method with output_dir
    import os
    import tempfile
    import shutil

    output_dir = tempfile.mkdtemp()
    result = signal_factor.eval(prices_factor, periods=1, quantiles=2, output_dir=output_dir)

    # Assertions
    assert isinstance(result, FactorAnalysisResult)
    # ic_summary is now always dict[int, dict[str, float]]
    assert 1 in result.ic_summary
    assert "mean_ic" in result.ic_summary[1]
    assert "ic_ir" in result.ic_summary[1]
    assert isinstance(result.ic_series, pd.DataFrame)
    assert isinstance(result.turnover_mean, float)
    # quantile_returns is now dict[int, pd.DataFrame]
    assert isinstance(result.quantile_returns, dict)
    assert 1 in result.quantile_returns
    assert len(result.quantile_returns[1].columns) == 2  # quantiles=2

    # Check if output directory was created and has files
    assert os.path.exists(output_dir)
    # Should have created a timestamped subdirectory
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    assert len(subdirs) == 1
    exp_dir = os.path.join(output_dir, subdirs[0])
    assert os.path.exists(os.path.join(exp_dir, "ic_series.csv"))
    assert os.path.exists(os.path.join(exp_dir, "plots"))

    # Cleanup
    shutil.rmtree(output_dir)
