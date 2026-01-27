"""Legacy engine implementations for testing purposes.

This module contains the old Pandas and Polars engine implementations
used for consistency testing between the two backends.
"""

from .pandas import PandasEngine
from .polars import PolarsEngine
from .protocol import ComputeEngine

_default_engine: ComputeEngine = PolarsEngine()


def get_engine() -> ComputeEngine:
    return _default_engine


def set_engine(engine: ComputeEngine) -> None:
    global _default_engine
    _default_engine = engine


__all__ = ["ComputeEngine", "PolarsEngine", "PandasEngine", "get_engine", "set_engine"]
