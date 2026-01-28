"""Utility functions for backtesting."""

import re
from typing import Union

import numpy as np
import pandas as pd
import polars as pl

from ..constants import (
    EPSILON,
    MAX_PERIODS_PER_YEAR,
    MIN_PERIODS_PER_YEAR,
    SECONDS_PER_YEAR,
)

# Backward compatibility alias
POSITION_EPSILON = EPSILON


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


def neutralize_weights_polars(
    df: pl.DataFrame, signal_col: str = "signal", group_col: str = "end_time"
) -> pl.DataFrame:
    """
    Neutralize weights to sum to zero (market neutral).

    Polars version for use in vectorized backtester.

    Args:
        df: DataFrame with signal column
        signal_col: Name of signal column
        group_col: Column to group by (usually timestamp)

    Returns:
        DataFrame with 'weight' column added
    """
    # Demean signal
    df = df.with_columns([(pl.col(signal_col) - pl.col(signal_col).mean().over(group_col)).alias("signal_demeaned")])

    # Normalize by sum of absolute values
    df = df.with_columns(
        [
            (pl.col("signal_demeaned") / pl.col("signal_demeaned").abs().sum().over(group_col))
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("weight")
        ]
    )

    return df.drop("signal_demeaned")


def safe_divide(
    a: Union[float, np.ndarray, pd.Series],
    b: Union[float, np.ndarray, pd.Series],
    default: float = np.nan,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Safe division that returns default when denominator is near zero.

    Uses EPSILON threshold per AGENTS.md safe_ function pattern.

    Args:
        a: Numerator
        b: Denominator
        default: Value to return if b is zero, NaN, or within EPSILON

    Returns:
        a/b, or default where |b| <= EPSILON
    """
    # Handle scalar
    if isinstance(b, (int, float, np.floating, np.integer)):
        if np.isnan(b) or abs(b) <= EPSILON:
            return default
        return a / b

    # Handle numpy array
    if isinstance(b, np.ndarray):
        # We need to handle 'a' potentially being an array or scalar too
        # np.where handles this correctly
        result = np.where(
            np.isnan(b) | (np.abs(b) <= EPSILON),
            default,
            a / np.where(np.abs(b) <= EPSILON, 1.0, b),  # Avoid divide by zero
        )
        return result

    # Handle pandas Series
    if isinstance(b, pd.Series):
        mask = b.isna() | (b.abs() <= EPSILON)
        result = a / b.where(~mask, 1.0)  # Replace near-zero with 1 to avoid error
        result = result.where(~mask, default)  # Then set result to default
        return result

    # Fallback
    try:
        if b == 0 or np.isnan(b):
            return default
    except (ValueError, TypeError):
        # Handle cases where b might be an array-like but not caught by above checks
        pass

    return a / b
