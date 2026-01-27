"""
Factor analysis module.

Exports:
    - Factor: Main factor class with time-series and math operations
    - BaseFactor: Base class for custom factor implementations
    - FactorExpressionParser: Parser for expression-based factor construction
    - PolarsEngine: High-performance computation engine using Polars
    - operators: Functional operators for factor expressions
"""

from .core import Factor
from .base import BaseFactor
from .parser import FactorExpressionParser
from .analyzer import FactorAnalyzer
from .engine import PolarsEngine

# Import all operators
from . import operators

__all__ = [
    "Factor",
    "BaseFactor",
    "FactorExpressionParser",
    "FactorAnalyzer",
    "PolarsEngine",
    "operators",
]
