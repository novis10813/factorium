from .base import BaseFactor
from typing import Union, Optional, TYPE_CHECKING
from pathlib import Path
import pandas as pd
from .mixins.math_ops import MathOpsMixin
from .mixins.ts_ops import TimeSeriesOpsMixin
from .mixins.cs_ops import CrossSectionalOpsMixin

if TYPE_CHECKING:
    from ..aggbar import AggBar


class Factor(CrossSectionalOpsMixin, TimeSeriesOpsMixin, MathOpsMixin, BaseFactor):
    """
    A factor representing a time-series of values for multiple symbols.
    
    Supports:
    - Arithmetic operations (+, -, *, /)
    - Comparison operations (<, <=, >, >=, ==, !=)
    - Time-series operations (ts_rank, ts_mean, ts_std, etc.)
    - Cross-sectional operations (rank, mean, median)
    - Mathematical operations (abs, log, pow, etc.)
    
    Example:
        >>> factor = Factor(data, name="close")
        >>> normalized = factor.ts_zscore(20)
        >>> ranked = normalized.rank()
    """
    
    def __init__(self, data: Union["AggBar", pd.DataFrame, Path], name: Optional[str] = None):
        super().__init__(data, name)
