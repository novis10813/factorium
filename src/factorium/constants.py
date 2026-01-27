"""Global constants for Factorium.

This module defines commonly used constants across the library, particularly
for numerical comparisons in factor calculations and backtesting.
"""

# Epsilon for numerical comparisons and safe division
# Used to detect when values are effectively zero
EPSILON = 1e-10

# Time-related constants
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

# Period frequency validation bounds
MIN_PERIODS_PER_YEAR = 1.0
MAX_PERIODS_PER_YEAR = 365.25 * 24 * 60  # Minutes in a year

__all__ = [
    "EPSILON",
    "SECONDS_PER_YEAR",
    "MIN_PERIODS_PER_YEAR",
    "MAX_PERIODS_PER_YEAR",
]
