"""ComputeEngine Protocol - defines the interface for computation backends."""

from typing import Protocol

import pandas as pd


class ComputeEngine(Protocol):
    """Protocol for computation engines.

    All engines must implement these methods to be compatible with Factor operations.
    Input/output is pd.DataFrame with standard columns:
    - start_time: int64 (milliseconds)
    - end_time: int64 (milliseconds)
    - symbol: str
    - factor: float64
    """

    # ==========================================
    # Time-series operations
    # ==========================================

    def ts_sum(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling sum with strict window."""
        ...

    def ts_mean(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling mean with strict window."""
        ...

    def ts_std(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling standard deviation with strict window."""
        ...

    def ts_min(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling minimum with strict window."""
        ...

    def ts_max(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling maximum with strict window."""
        ...

    def ts_median(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling median with strict window."""
        ...

    def ts_product(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling product with strict window."""
        ...

    def ts_rank(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling percentile rank with strict window."""
        ...

    def ts_argmin(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Position of minimum value from current position."""
        ...

    def ts_argmax(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Position of maximum value from current position."""
        ...

    def ts_shift(
        self,
        df: pd.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Shift values by period."""
        ...

    def ts_diff(
        self,
        df: pd.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Difference from period ago."""
        ...

    def ts_kurtosis(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling kurtosis with strict window."""
        ...

    def ts_corr(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling correlation between two factors."""
        ...

    def ts_cov(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling covariance between two factors."""
        ...

    def ts_cv(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Rolling coefficient of variation."""
        ...

    def ts_reversal_count(
        self,
        df: pd.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Count of sign reversals in window."""
        ...

    # ==========================================
    # Cross-sectional operations
    # ==========================================

    def cs_rank(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Cross-sectional percentile rank."""
        ...

    def cs_zscore(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Cross-sectional z-score standardization."""
        ...

    def cs_demean(
        self,
        df: pd.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Cross-sectional de-meaning."""
        ...

    def cs_winsorize(
        self,
        df: pd.DataFrame,
        lower_limit: float,
        upper_limit: float,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Cross-sectional winsorization."""
        ...

    # ==========================================
    # Element-wise math operations
    # ==========================================

    def abs(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        """Element-wise absolute value."""
        ...

    def sign(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        """Element-wise sign."""
        ...

    def log(self, df: pd.DataFrame, base: float | None = None, value_col: str = "factor") -> pd.DataFrame:
        """Element-wise logarithm."""
        ...

    def sqrt(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        """Element-wise square root."""
        ...

    def pow(self, df: pd.DataFrame, exponent: float, value_col: str = "factor") -> pd.DataFrame:
        """Element-wise power."""
        ...

    def neg(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        """Element-wise negation."""
        ...

    def inverse(self, df: pd.DataFrame, value_col: str = "factor") -> pd.DataFrame:
        """Element-wise inverse (1/x)."""
        ...

    # ==========================================
    # Binary math operations
    # ==========================================

    def signed_pow(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Signed power with another factor."""
        ...

    def maximum(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Element-wise maximum of two factors."""
        ...

    def minimum(
        self,
        df: pd.DataFrame,
        other_df: pd.DataFrame,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Element-wise minimum of two factors."""
        ...

    def where(
        self,
        df: pd.DataFrame,
        cond_df: pd.DataFrame,
        other_df: pd.DataFrame | None,
        other_scalar: float | None = None,
        value_col: str = "factor",
    ) -> pd.DataFrame:
        """Conditional selection."""
        ...
