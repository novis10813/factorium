# pytest configuration for factorium tests
import os
import sys

# Add the project src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pandas as pd
import pytest

from factorium import AggBar, Factor


@pytest.fixture(scope="module")
def sample_aggbar():
    """
    Creates a synthetic AggBar with BTCUSDT and ETHUSDT data.
    Includes positive, negative, and zero values to test various math edge cases.
    """
    dates = pd.date_range(start="2025-01-01", periods=10, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6

    common_cols = {
        "start_time": timestamps,
        "end_time": timestamps + 60000,
    }

    df_btc_processed = pd.DataFrame(common_cols)
    df_btc_processed["symbol"] = "BTCUSDT"
    df_btc_processed["close"] = [100.0, -50.0, 0.0, 25.0, 1000.0, -1000.0, 0.0001, 10.0, 50.0, 100.0]
    df_btc_processed["open"] = [50.0, 50.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    df_eth_processed = pd.DataFrame(common_cols)
    df_eth_processed["symbol"] = "ETHUSDT"
    df_eth_processed["close"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    df_eth_processed["open"] = [5.0] * 10

    df_combined = pd.concat([df_btc_processed, df_eth_processed], ignore_index=True)

    return AggBar(df_combined)


@pytest.fixture
def factor_close(sample_aggbar):
    return sample_aggbar["close"]


@pytest.fixture
def factor_open(sample_aggbar):
    return sample_aggbar["open"]
