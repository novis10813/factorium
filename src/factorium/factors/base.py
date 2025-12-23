from typing import Union, Optional, Callable, Self, TYPE_CHECKING
from abc import ABC
import pandas as pd
from pathlib import Path
import numpy as np

if TYPE_CHECKING:
    from ..aggbar import AggBar


class BaseFactor(ABC):
    def __init__(self, data: Union["AggBar", pd.DataFrame, Path], name: Optional[str] = None):
        self._name = name or 'factor'
        
        if isinstance(data, Path):
            if data.suffix == '.csv':
                self._data = pd.read_csv(data)
            elif data.suffix == '.parquet':
                self._data = pd.read_parquet(data)
            else:
                raise ValueError(f"Invalid file extension: {data.suffix}")
        elif hasattr(data, 'to_df'):  # AggBar-like object
            self._data = data.to_df()
        elif isinstance(data, pd.DataFrame):
            self._data = data.copy()
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        
        if isinstance(self._data.index, pd.MultiIndex):
            self._data = self._data.reset_index()
            
        if len(self._data.columns) == 4 and 'factor' not in self._data.columns:
            self._data.columns = ["start_time", "end_time", "symbol", "factor"]
            
        elif "factor" not in self._data.columns:
            factor_columns = [col for col in self._data.columns if col not in ["start_time", "end_time", "symbol"]]
            
            if not factor_columns:
                raise ValueError("No factor columns found")
            
            self._data = self._data[["start_time", "end_time", "symbol", factor_columns[0]]]
            
        self._data = self._data.sort_values(by=['end_time', 'symbol']).reset_index(drop=True)
        
        self._data.columns = ['start_time', 'end_time', 'symbol', 'factor']
        
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str):
        self._name = name
        
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    def _validate_window(self, window: int) -> None:
        if window <= 0:
            raise ValueError("Window must be positive")
    
    def _validate_factor(self, other: Self, op_name: str) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError(f"{op_name}: other must be a Factor object")
        
    @staticmethod
    def _replace_inf(series: pd.Series) -> pd.Series:
        return series.replace([np.inf, -np.inf], np.nan)
        
    def _binary_op(self, other: Union['BaseFactor', float], op_func: Callable, 
                   op_name: str, scalar_suffix: Optional[str] = None) -> Self:
        if isinstance(other, self.__class__):
            merged = pd.merge(self._data, other._data,
                            on=['start_time', 'end_time', 'symbol'],
                            suffixes=('_x', '_y'),
                            how='inner')
            merged['factor'] = op_func(merged['factor_x'], merged['factor_y'])
            result = merged[['start_time', 'end_time', 'symbol', 'factor']]
            return self.__class__(result, f"({self.name}{op_name}{other.name})")
        else:
            result = self._data.copy()
            result['factor'] = op_func(result['factor'], other)
            suffix = scalar_suffix if scalar_suffix is not None else str(other)
            return self.__class__(result, f"({self.name}{op_name}{suffix})")
        
    def _comparison_op(self, other: Union['BaseFactor', float], comp_func: Callable, 
                       op_name: str) -> Self:
        if isinstance(other, self.__class__):
            merged = pd.merge(self._data, other._data,
                            on=['start_time', 'end_time', 'symbol'],
                            suffixes=('_x', '_y'))
            merged['factor'] = comp_func(merged['factor_x'], merged['factor_y']).astype(int)
            result = merged[['start_time', 'end_time', 'symbol', 'factor']]
        else:
            result = self._data.copy()
            result['factor'] = comp_func(result['factor'], other).astype(int)
        return self.__class__(result, f"({self.name}{op_name}{getattr(other, 'name', other)})")
    
    def _cs_op(self, operation: Callable, name_suffix: str,  require_no_nan: bool = False) -> Self:
        result = self._data.copy()
        result['factor'] = pd.to_numeric(result['factor'], errors='coerce')
        
        if require_no_nan and result['factor'].isna().all():
            raise ValueError("All factor values are NaN")
        
        def safe_op(group):
            if group.isna().any():
                return pd.Series(np.nan, index=group.index)
            output = operation(group)
            if isinstance(output, (int, float, np.number)):
                return pd.Series(output, index=group.index)
            return output
        
        result['factor'] = result.groupby('end_time')['factor'].transform(safe_op)
        return self.__class__(result, f"{name_suffix}({self.name})")
    
    def _apply_rolling(self, func: Union[Callable, str], window: int) -> pd.DataFrame:
        result = self._data.copy()
        
        if isinstance(func, str):
            result['factor'] = (result.groupby('symbol')['factor']
                                .rolling(window=window, min_periods=window)
                                .agg(func)
                                .reset_index(level=0, drop=True))
        
        else:
            result['factor'] = (result.groupby('symbol')['factor']
                                .rolling(window, min_periods=window)
                                .apply(func, raw=False)
                                .reset_index(level=0, drop=True))
        return result
    
    def __mul__(self, other: Union['BaseFactor', float]) -> Self:
        return self._binary_op(other, lambda x, y: x * y, '*')
    
    def __neg__(self) -> Self:
        return self.__mul__(-1)
    
    def __add__(self, other: Union['BaseFactor', float]) -> Self:
        return self._binary_op(other, lambda x, y: x + y, '+')
    
    def __sub__(self, other: Union['BaseFactor', float]) -> Self:
        return self._binary_op(other, lambda x, y: x - y, '-')
    
    def __truediv__(self, other: Union['BaseFactor', float]) -> Self:
        def safe_div(x, y):
            if isinstance(y, (int, float)):
                if y == 0:
                    return np.nan
                return x / y
            else:
                return np.where(np.abs(y) > 1e-10, x / y, np.nan)
        return self._binary_op(other, safe_div, '/')
    
    def __rtruediv__(self, other: Union['BaseFactor', float]) -> Self:
        if isinstance(other, self.__class__):
            return other.__truediv__(self)
        else:
            result = self._data.copy()
            result['factor'] = np.where(
                result['factor'] != 0,
                other / result['factor'],
                np.nan
            )
            return self.__class__(result, f"({other}/{self.name})")
    
    def __lt__(self, other: Union['BaseFactor', float]) -> Self:
        return self._comparison_op(other, lambda x, y: x < y, '<')
    
    def __le__(self, other: Union['BaseFactor', float]) -> Self:
        return self._comparison_op(other, lambda x, y: x <= y, '<=')
    
    def __gt__(self, other: Union['BaseFactor', float]) -> Self:
        return self._comparison_op(other, lambda x, y: x > y, '>')
    
    def __ge__(self, other: Union['BaseFactor', float]) -> Self:
        return self._comparison_op(other, lambda x, y: x >= y, '>=')
    
    def __eq__(self, other: Union['BaseFactor', float]) -> Self:
        return self._comparison_op(other, lambda x, y: x == y, '==')
    
    def __ne__(self, other: Union['BaseFactor', float]) -> Self:
        return self._comparison_op(other, lambda x, y: x != y, '!=')
    
    def __len__(self) -> int:
        return len(self._data)
