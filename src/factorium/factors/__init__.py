"""
Factor analysis module.

Exports:
    - Factor: Main factor class with time-series and math operations
    - BaseFactor: Base class for custom factor implementations
    - FactorExpressionParser: Parser for expression-based factor construction
    - PolarsEngine: High-performance computation engine using Polars
    - operators: Functional operators for factor expressions
"""

# Import all operators
from . import operators
from .analyzer import FactorAnalysisResult, FactorAnalyzer
from .base import BaseFactor
from .composite import CompositeFactor
from .core import Factor
from .engine import PolarsEngine
from .parser import FactorExpressionParser

__all__ = [
    "Factor",
    "BaseFactor",
    "FactorExpressionParser",
    "FactorAnalyzer",
    "FactorAnalysisResult",
    "PolarsEngine",
    "CompositeFactor",
    "operators",
]
