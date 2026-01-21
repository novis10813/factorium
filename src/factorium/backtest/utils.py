"""Utility functions for backtesting."""

import re

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

POSITION_EPSILON = 1e-10
MIN_PERIODS_PER_YEAR = 1.0
MAX_PERIODS_PER_YEAR = 365.25 * 24 * 60


def parse_frequency_to_seconds(freq: str) -> float:
    """
    Parse pandas-style frequency string to seconds.

    Supports: s (seconds), m (minutes), h (hours), d (days), w (weeks)

    Examples:
        "1h" -> 3600
        "30m" -> 1800
        "1d" -> 86400
    """
    match = re.match(r"^(\d+)([smhdw])$", freq.lower())
    if not match:
        raise ValueError(f"Invalid frequency format: '{freq}'. Use format like '1h', '30m', '1d'")

    value = int(match.group(1))
    unit = match.group(2)

    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }

    return float(value * multipliers[unit])


def frequency_to_periods_per_year(freq: str) -> float:
    seconds = parse_frequency_to_seconds(freq)
    return SECONDS_PER_YEAR / seconds


def neutralize_weights(signals: pd.Series) -> pd.Series:
    """
    Convert signals to dollar-neutral weights.

    Formula: (x - mean) / sum(|x - mean|)

    This ensures:
    1. Long and short weights sum to 0
    2. Total absolute weight equals 1

    Args:
        signals: Raw signal values indexed by symbol

    Returns:
        Neutralized weights (sum to 0, abs sum to 1)

    Example:
        >>> signals = pd.Series([0.8, 0.5, 0.3, 0.1], index=['A', 'B', 'C', 'D'])
        >>> weights = neutralize_weights(signals)
        >>> abs(weights.sum()) < 1e-10  # Sum to 0
        True
        >>> abs(weights.abs().sum() - 1.0) < 1e-10  # Abs sum to 1
        True
    """
    signals = signals.dropna()

    if len(signals) == 0:
        return pd.Series(dtype=float)

    mean = signals.mean()
    demeaned = signals - mean

    abs_sum = demeaned.abs().sum()
    if abs_sum == 0:
        return pd.Series(0.0, index=signals.index)

    return demeaned / abs_sum


def normalize_weights(signals: pd.Series) -> pd.Series:
    """
    Normalize positive signals to weights that sum to 1 (long-only).

    Note:
        Negative and zero signals are filtered out before normalization.
        This is intended for long-only strategies where only positive
        signals indicate buy interest.

    Args:
        signals: Raw signal values indexed by symbol

    Returns:
        Normalized weights (sum to 1, all positive). Empty Series if no
        positive signals exist.
    """
    valid_signals = signals.dropna()
    positive_signals = valid_signals[valid_signals > 0]

    if len(positive_signals) == 0:
        return pd.Series(dtype=float)

    total = positive_signals.sum()
    if total == 0:
        return pd.Series(0.0, index=positive_signals.index)

    return pd.Series(positive_signals / total)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safe division that avoids division by zero.

    Args:
        a: Numerator
        b: Denominator
        default: Value to return if b is zero or NaN

    Returns:
        a / b if b is valid, else default
    """
    if b == 0 or np.isnan(b):
        return default
    return a / b
