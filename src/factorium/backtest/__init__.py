from .backtester import (
    IterativeBacktester as LegacyBacktester,
)
from .backtester import (
    IterativeBacktestResult as LegacyBacktestResult,
)
from .constraints import (
    LongOnlyConstraint,
    MarketNeutralConstraint,
    MaxGrossExposureConstraint,
    MaxPositionConstraint,
    WeightConstraint,
)
from .metrics import calculate_metrics
from .portfolio import Portfolio
from .utils import (
    MAX_PERIODS_PER_YEAR,
    MIN_PERIODS_PER_YEAR,
    POSITION_EPSILON,
    frequency_to_periods_per_year,
    neutralize_weights,
    normalize_weights,
    parse_frequency_to_seconds,
)
from .vectorized import BacktestResult, VectorizedBacktester

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
