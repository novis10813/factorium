from .backtester import (
    IterativeBacktester as LegacyBacktester,
    IterativeBacktestResult as LegacyBacktestResult,
)
from .metrics import calculate_metrics
from .portfolio import Portfolio
from .vectorized import VectorizedBacktester, BacktestResult
from .constraints import (
    WeightConstraint,
    MaxPositionConstraint,
    LongOnlyConstraint,
    MaxGrossExposureConstraint,
    MarketNeutralConstraint,
)
from .utils import (
    MAX_PERIODS_PER_YEAR,
    MIN_PERIODS_PER_YEAR,
    POSITION_EPSILON,
    frequency_to_periods_per_year,
    neutralize_weights,
    normalize_weights,
    parse_frequency_to_seconds,
)

# Backward compatibility: Backtester is now an alias for VectorizedBacktester
Backtester = VectorizedBacktester

__all__ = [
    "Backtester",
    "LegacyBacktester",
    "BacktestResult",
    "LegacyBacktestResult",
    "VectorizedBacktester",
    "Portfolio",
    "calculate_metrics",
    "WeightConstraint",
    "MaxPositionConstraint",
    "LongOnlyConstraint",
    "MaxGrossExposureConstraint",
    "MarketNeutralConstraint",
    "frequency_to_periods_per_year",
    "neutralize_weights",
    "normalize_weights",
    "parse_frequency_to_seconds",
    "POSITION_EPSILON",
    "MIN_PERIODS_PER_YEAR",
    "MAX_PERIODS_PER_YEAR",
]
