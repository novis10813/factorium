import numpy as np
import pandas as pd
import pytest
from factorium.backtest.utils import safe_divide
from factorium.constants import EPSILON


def test_safe_divide_uses_epsilon():
    """safe_divide should use EPSILON threshold."""
    # Exact zero -> NaN
    assert np.isnan(safe_divide(1.0, 0.0))

    # Near zero (within EPSILON) -> NaN
    assert np.isnan(safe_divide(1.0, EPSILON / 2))
    assert np.isnan(safe_divide(1.0, -EPSILON / 2))

    # Just above EPSILON -> valid division
    result = safe_divide(1.0, EPSILON * 2)
    assert not np.isnan(result)

    # Normal case
    assert safe_divide(10.0, 2.0) == 5.0

    # Default value
    assert safe_divide(1.0, 0.0, default=0.0) == 0.0


def test_safe_divide_handles_arrays():
    """safe_divide should work with numpy arrays."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 0.0, 1.0])

    result = safe_divide(a, b)

    assert result[0] == 0.5
    assert np.isnan(result[1])
    assert result[2] == 3.0


def test_safe_divide_handles_series():
    """safe_divide should work with pandas Series."""
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([2.0, 0.0, 1.0])

    result = safe_divide(a, b)

    assert result[0] == 0.5
    assert np.isnan(result[1])
    assert result[2] == 3.0
