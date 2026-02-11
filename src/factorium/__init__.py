"""
Factorium - A quantitative factor analysis library.

Provides tools for building and analyzing financial factors with support for:
- Time-series operations (ts_rank, ts_mean, ts_std, etc.)
- Cross-sectional operations (rank, mean, median)
- Mathematical operations (abs, log, pow, etc.)
- Data loading from Binance Vision

Usage:
    from factorium import Factor, AggBar
    from factorium import BinanceDataLoader

Example:
    >>> from factorium import AggBar, BinanceDataLoader
    >>> loader = BinanceDataLoader()
    >>> agg = loader.load_aggbar(
    ...     symbols=["BTCUSDT"],
    ...     data_type="aggTrades",
    ...     market_type="futures",
    ...     futures_type="um",
    ...     start_date="2024-01-01",
    ...     days=7,
    ...     bar_type="time",
    ...     interval=60_000,
    ... )
    >>> close = agg['close']
    >>> momentum = close.ts_delta(20) / close.ts_shift(20)
    >>> ranked = momentum.rank()
"""

from .factors.core import Factor
from .factors.base import BaseFactor
from .aggbar import AggBar
from .data import BinanceDataLoader
from .research import ResearchSession

__version__ = "0.2.1"

__all__ = [
    # Core classes
    "Factor",
    "BaseFactor",
    "AggBar",
    "ResearchSession",
    # Data loading
    "BinanceDataLoader",
]
