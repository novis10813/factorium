"""
Factor analysis module.

Exports:
    - Factor: Main factor class with time-series and math operations
    - BaseFactor: Base class for custom factor implementations
"""

from .core import Factor
from .base import BaseFactor

__all__ = ["Factor", "BaseFactor"]
