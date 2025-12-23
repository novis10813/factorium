import pandas as pd
import numpy as np
from typing import Self, Optional, Union


class MathOpsMixin:
    def abs(self) -> Self:
        result = self._data.copy()
        result['factor'] = np.abs(result['factor'])
        return self.__class__(result, f"abs({self.name})")
    
    def sign(self) -> Self:
        result = self._data.copy()
        result['factor'] = np.sign(result['factor'])
        return self.__class__(result, f"sign({self.name})")
    
    def inverse(self) -> Self:
        result = self._data.copy()
        result['factor'] = np.where(
            result['factor'] != 0,
            1 / result['factor'],
            np.nan
        )
        return self.__class__(result, f"inverse({self.name})")
    
    def log(self, base: Optional[float] = None) -> Self:
        result = self.data.copy()
        vals = result['factor']
        mask = vals > 0

        if base is None:
            log_vals = np.log(vals[mask])
            name = f"log({self.name})"
        else:
            if base <= 0 or base == 1:
                raise ValueError(f"Invalid log base: {base}. Base must be greater than 0 and not equal to 1.")
            log_vals = np.log(vals[mask]) / np.log(base)
            name = f"log({self.name},base={base})"

        result['factor'] = np.nan
        result.loc[mask, 'factor'] = log_vals
        return self.__class__(result, name)

    def ln(self) -> Self:
        return self.log()
    
    def sqrt(self) -> Self:
        result = self._data.copy()
        result['factor'] = np.where(
            result['factor'] > 0,
            np.sqrt(result['factor']),
            np.nan
        )
        return self.__class__(result, f"sqrt({self.name})")
    
    def signed_log1p(self) -> Self:
        result = self._data.copy()
        result['factor'] = np.sign(result['factor']) * np.log1p(np.abs(result['factor']))
        return self.__class__(result, f"signed_log1p({self.name})")
    
    def signed_pow(self, exponent: Union[Self, float]) -> Self:
        if isinstance(exponent, self.__class__):
            merged = pd.merge(
                self._data, exponent.data,
                on=['start_time', 'end_time', 'symbol'],
                suffixes=('_x', '_y')
            )
            
            sign = np.sign(merged['factor_x'])
            abs_val = np.abs(merged['factor_x'])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                result_val = sign * (abs_val ** merged['factor_y'])
                
            merged['factor'] = self._replace_inf(result_val)
            
            result = merged[['start_time', 'end_time', 'symbol', 'factor']]
        
        else:
            result = self._data.copy()
            
            sign = np.sign(result['factor'])
            abs_val = np.abs(result['factor'])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                result_val = sign * (abs_val ** exponent)
                
            result['factor'] = self._replace_inf(result_val)
            
        return self.__class__(result, f"signed_pow({self.name},{exponent})")
    
    def pow(self, exponent: Union[Self, float]) -> Self:
        if isinstance(exponent, self.__class__):
            merged = pd.merge(
                self._data, exponent.data,
                on=['start_time', 'end_time', 'symbol'],
                suffixes=('_x', '_y')
            )
            
            with np.errstate(divide='ignore', invalid='ignore'):
                merged['factor'] = merged['factor_x'] ** merged['factor_y']
                
            merged['factor'] = self._replace_inf(merged['factor'])
            
            result = merged[['start_time', 'end_time', 'symbol', 'factor']]
        else:
            result = self._data.copy()
            with np.errstate(divide='ignore', invalid='ignore'):
                result['factor'] = result['factor'] ** exponent
                
            result['factor'] = self._replace_inf(result['factor'])
        return self.__class__(result, f"pow({self.name},{exponent})")
    
    def add(self, other: Union[Self, float]) -> Self:
        return self.__add__(other)
    
    def sub(self, other: Union[Self, float]) -> Self:
        return self.__sub__(other)
    
    def mul(self, other: Union[Self, float]) -> Self:
        return self.__mul__(other)
    
    def div(self, other: Union[Self, float]) -> Self:
        return self.__truediv__(other)
    
    def where(self, cond: Self, other: Union[Self, float] = np.nan) -> Self:
        if not isinstance(cond, self.__class__):
            raise ValueError(f"Condition must be a Factor, got {type(cond)}")
        
        merged = pd.merge(
            self._data, cond.data,
            on=['start_time', 'end_time', 'symbol'],
            suffixes=('', '_cond')
        )
        
        cond_bool = merged['factor_cond'].fillna(False).astype(bool)        
        
        if isinstance(other, self.__class__):
            merged = pd.merge(merged, other.data.rename(columns={'factor': 'factor_other'}), on=['start_time', 'end_time', 'symbol'])
            
            merged['factor'] = np.where(cond_bool, merged['factor'], merged['factor_other'])
            
        else:
            merged['factor'] = np.where(cond_bool, merged['factor'], other)
            
        result = merged[['start_time', 'end_time', 'symbol', 'factor']]
        return self.__class__(result, f"where({self.name})")
    
    def max(self, other: Union[Self, float]) -> Self:
        if isinstance(other, self.__class__):
            merged = pd.merge(
                self._data, other.data,
                on=['start_time', 'end_time', 'symbol'],
                suffixes=('_x', '_y')
            )
            
            merged['factor'] = np.maximum(merged['factor_x'], merged['factor_y'])
            
            result = merged[['start_time', 'end_time', 'symbol', 'factor']]
        else:
            result = self._data.copy()
            result['factor'] = np.maximum(result['factor'], other)
        return self.__class__(result, f"max({self.name},{other})")
    
    def min(self, other: Union[Self, float]) -> Self:
        if isinstance(other, self.__class__):
            merged = pd.merge(
                self._data, other.data,
                on=['start_time', 'end_time', 'symbol'],
                suffixes=('_x', '_y')
            )
            merged['factor'] = np.minimum(merged['factor_x'], merged['factor_y'])
            result = merged[['start_time', 'end_time', 'symbol', 'factor']]
        else:
            result = self._data.copy()
            result['factor'] = np.minimum(result['factor'], other)
        return self.__class__(result, f"min({self.name},{other})")


    def reverse(self) -> Self:
        return self.__neg__()
