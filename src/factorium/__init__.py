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
    >>> df = loader.load_data(symbol="BTCUSDT", data_type="aggTrades", ...)
    >>> agg = AggBar(df)
    >>> close = agg['close']
    >>> momentum = close.ts_delta(20) / close.ts_shift(20)
    >>> ranked = momentum.rank()
"""

from .factors.core import Factor
from .factors.base import BaseFactor
from .aggbar import AggBar
from .data import BinanceDataLoader

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Factor",
    "BaseFactor",
    "AggBar",
    # Data loading
    "BinanceDataLoader",
]
