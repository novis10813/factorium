"""
Functional operators for Factor expressions.

This module provides functional-style wrappers for all Factor operations,
enabling expression-based factor construction similar to alpha101.
"""

from typing import Union, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .core import Factor


# ============================================================================
# Time Series Operators
# ============================================================================

def ts_rank(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_rank(window)"""
    return factor.ts_rank(window)


def ts_sum(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_sum(window)"""
    return factor.ts_sum(window)


def ts_product(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_product(window)"""
    return factor.ts_product(window)


def ts_mean(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_mean(window)"""
    return factor.ts_mean(window)


def ts_median(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_median(window)"""
    return factor.ts_median(window)


def ts_std(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_std(window)"""
    return factor.ts_std(window)


def ts_min(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_min(window)"""
    return factor.ts_min(window)


def ts_max(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_max(window)"""
    return factor.ts_max(window)


def ts_argmin(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_argmin(window)"""
    return factor.ts_argmin(window)


def ts_argmax(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_argmax(window)"""
    return factor.ts_argmax(window)


def ts_scale(factor: "Factor", window: int, constant: float = 0) -> "Factor":
    """Functional version of factor.ts_scale(window, constant)"""
    return factor.ts_scale(window, constant)


def ts_zscore(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_zscore(window)"""
    return factor.ts_zscore(window)


def ts_quantile(factor: "Factor", window: int, driver: str = "gaussian") -> "Factor":
    """Functional version of factor.ts_quantile(window, driver)"""
    return factor.ts_quantile(window, driver)


def ts_kurtosis(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_kurtosis(window)"""
    return factor.ts_kurtosis(window)


def ts_skewness(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_skewness(window)"""
    return factor.ts_skewness(window)


def ts_step(factor: "Factor", start: int = 1) -> "Factor":
    """Functional version of factor.ts_step(start)"""
    return factor.ts_step(start)


def ts_shift(factor: "Factor", period: int) -> "Factor":
    """Functional version of factor.ts_shift(period)"""
    return factor.ts_shift(period)


def ts_delta(factor: "Factor", period: int) -> "Factor":
    """Functional version of factor.ts_delta(period)"""
    return factor.ts_delta(period)


def ts_corr(factor1: "Factor", factor2: "Factor", window: int) -> "Factor":
    """Functional version of factor1.ts_corr(factor2, window)"""
    return factor1.ts_corr(factor2, window)


def ts_cov(factor1: "Factor", factor2: "Factor", window: int) -> "Factor":
    """Functional version of factor1.ts_cov(factor2, window)"""
    return factor1.ts_cov(factor2, window)


def ts_cv(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_cv(window)"""
    return factor.ts_cv(window)


def ts_jumpiness(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_jumpiness(window)"""
    return factor.ts_jumpiness(window)


def ts_autocorr(factor: "Factor", window: int, lag: int = 1) -> "Factor":
    """Functional version of factor.ts_autocorr(window, lag)"""
    return factor.ts_autocorr(window, lag)


def ts_reversal_count(factor: "Factor", window: int) -> "Factor":
    """Functional version of factor.ts_reversal_count(window)"""
    return factor.ts_reversal_count(window)


def ts_vr(factor: "Factor", window: int, k: int = 2) -> "Factor":
    """Functional version of factor.ts_vr(window, k)"""
    return factor.ts_vr(window, k)


# ============================================================================
# Cross-Sectional Operators
# ============================================================================

def rank(factor: "Factor") -> "Factor":
    """Functional version of factor.rank()"""
    return factor.rank()


def mean(factor: "Factor") -> "Factor":
    """Functional version of factor.mean()"""
    return factor.mean()


def median(factor: "Factor") -> "Factor":
    """Functional version of factor.median()"""
    return factor.median()


# ============================================================================
# Math Operators
# ============================================================================

def abs(factor: "Factor") -> "Factor":
    """Functional version of factor.abs()"""
    return factor.abs()


def sign(factor: "Factor") -> "Factor":
    """Functional version of factor.sign()"""
    return factor.sign()


def inverse(factor: "Factor") -> "Factor":
    """Functional version of factor.inverse()"""
    return factor.inverse()


def log(factor: "Factor", base: Optional[float] = None) -> "Factor":
    """Functional version of factor.log(base)"""
    return factor.log(base)


def ln(factor: "Factor") -> "Factor":
    """Functional version of factor.ln()"""
    return factor.ln()


def sqrt(factor: "Factor") -> "Factor":
    """Functional version of factor.sqrt()"""
    return factor.sqrt()


def signed_log1p(factor: "Factor") -> "Factor":
    """Functional version of factor.signed_log1p()"""
    return factor.signed_log1p()


def signed_pow(factor: "Factor", exponent: Union["Factor", float]) -> "Factor":
    """Functional version of factor.signed_pow(exponent)"""
    return factor.signed_pow(exponent)


def pow(factor: "Factor", exponent: Union["Factor", float]) -> "Factor":
    """Functional version of factor.pow(exponent)"""
    return factor.pow(exponent)


def where(factor: "Factor", cond: "Factor", other: Union["Factor", float] = None) -> "Factor":
    """Functional version of factor.where(cond, other)"""
    import numpy as np
    if other is None:
        other = np.nan
    return factor.where(cond, other)


def max(factor: "Factor", other: Union["Factor", float]) -> "Factor":
    """Functional version of factor.max(other)"""
    return factor.max(other)


def min(factor: "Factor", other: Union["Factor", float]) -> "Factor":
    """Functional version of factor.min(other)"""
    return factor.min(other)


def reverse(factor: "Factor") -> "Factor":
    """Functional version of factor.reverse()"""
    return factor.reverse()


# ============================================================================
# Binary Operators
# ============================================================================

def add(factor1: Union["Factor", float], factor2: Union["Factor", float]) -> "Factor":
    """Functional version of factor1 + factor2"""
    if isinstance(factor1, float) or isinstance(factor1, int):
        # Handle scalar on left side
        return factor2 + factor1
    return factor1 + factor2


def sub(factor1: Union["Factor", float], factor2: Union["Factor", float]) -> "Factor":
    """Functional version of factor1 - factor2"""
    if isinstance(factor1, float) or isinstance(factor1, int):
        # Handle scalar on left side
        from .core import Factor
        if isinstance(factor2, Factor):
            result = factor2._data.copy()
            result['factor'] = factor1 - result['factor']
            return Factor(result, f"({factor1}-{factor2.name})")
        return factor1 - factor2
    return factor1 - factor2


def mul(factor1: Union["Factor", float], factor2: Union["Factor", float]) -> "Factor":
    """Functional version of factor1 * factor2"""
    if isinstance(factor1, float) or isinstance(factor1, int):
        return factor2 * factor1
    return factor1 * factor2


def div(factor1: Union["Factor", float], factor2: Union["Factor", float]) -> "Factor":
    """Functional version of factor1 / factor2"""
    if isinstance(factor1, float) or isinstance(factor1, int):
        # Handle scalar on left side
        from .core import Factor
        if isinstance(factor2, Factor):
            result = factor2._data.copy()
            result['factor'] = np.where(
                result['factor'] != 0,
                factor1 / result['factor'],
                np.nan
            )
            return Factor(result, f"({factor1}/{factor2.name})")
        return factor1 / factor2
    return factor1 / factor2

