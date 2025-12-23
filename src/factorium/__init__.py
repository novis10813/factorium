"""
Factorium - A quantitative factor analysis library.

Provides tools for building and analyzing financial factors with support for:
- Time-series operations (ts_rank, ts_mean, ts_std, etc.)
- Cross-sectional operations (rank, mean, median)
- Mathematical operations (abs, log, pow, etc.)
- Multiple bar sampling methods (time, tick, volume, dollar)
- Data loading from Binance Vision

Usage:
    from factorium import Factor, AggBar, TimeBar, TickBar, VolumeBar, DollarBar
    from factorium import BinanceDataLoader

Example:
    >>> from factorium import AggBar, TimeBar, BinanceDataLoader
    >>> loader = BinanceDataLoader()
    >>> df = loader.load_data(symbol="BTCUSDT", data_type="aggTrades", ...)
    >>> bar = TimeBar(df, interval_ms=60_000)
    >>> agg = AggBar([bar])
    >>> close = agg['close']
    >>> momentum = close.ts_delta(20) / close.ts_shift(20)
    >>> ranked = momentum.rank()
"""

from .factors.core import Factor
from .factors.base import BaseFactor
from .aggbar import AggBar
from .bar import BaseBar, TimeBar, TickBar, VolumeBar, DollarBar
from .data_loader import BinanceDataLoader

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Factor",
    "BaseFactor",
    "AggBar",
    # Bar types
    "BaseBar",
    "TimeBar",
    "TickBar",
    "VolumeBar",
    "DollarBar",
    # Data loading
    "BinanceDataLoader",
]
