import pandas as pd
import numpy as np
from typing import Self
from scipy.stats import norm, uniform, cauchy


class TimeSeriesOpsMixin:
    def ts_rank(self, window: int) -> Self:
        self._validate_window(window)
        
        result = self._data.copy()
        
        def rank_window(w):
            if np.isnan(w).any() or len(np.unique(w)) == 1:
                return np.nan
            sorted_idx = np.argsort(w)
            rank_array = np.empty_like(sorted_idx, dtype=float)
            rank_array[sorted_idx] = np.arange(1, len(w) + 1)
            return rank_array[-1] / len(w)
        
        result['factor'] = result.groupby('symbol')['factor'].rolling(window, min_periods=window).apply(rank_window, raw=True).reset_index(level=0, drop=True)
        
        return self.__class__(result, f"ts_rank({self.name},{window})")
    
    def ts_sum(self, window: int) -> Self:
        self._validate_window(window)
        
        result = self._data.copy()
        
        def safe_sum(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.sum()
        
        result = self._apply_rolling(safe_sum, window)
        
        return self.__class__(result, f"ts_sum({self.name},{window})")
    
    def ts_product(self, window: int) -> Self:
        self._validate_window(window)
        
        def safe_prod(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.prod()
        
        result = self._apply_rolling(safe_prod, window)
        
        return self.__class__(result, f"ts_product({self.name},{window})")
    
    def ts_mean(self, window: int) -> Self:
        self._validate_window(window)
        
        def safe_mean(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.mean()
        
        result = self._apply_rolling(safe_mean, window)
        
        return self.__class__(result, f"ts_mean({self.name},{window})")
    
    def ts_median(self, window: int) -> Self:
        self._validate_window(window)
        
        def safe_median(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.median()
        
        result = self._apply_rolling(safe_median, window)
        
        return self.__class__(result, f"ts_median({self.name},{window})")
    
    def ts_std(self, window: int) -> Self:
        self._validate_window(window)
        
        def safe_std(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.std()
        
        result = self._apply_rolling(safe_std, window)
        
        return self.__class__(result, f"ts_std({self.name},{window})")
    
    def ts_min(self, window: int) -> Self:
        self._validate_window(window)
        
        def safe_min(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.min()
        
        result = self._apply_rolling(safe_min, window)
        
        return self.__class__(result, f"ts_min({self.name},{window})")
    
    def ts_max(self, window: int) -> Self:
        self._validate_window(window)
        
        def safe_max(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else x.max()
        
        result = self._apply_rolling(safe_max, window)
        
        return self.__class__(result, f"ts_max({self.name},{window})")
    
    def ts_argmin(self, window: int) -> Self:
        self._validate_window(window)
        
        def safe_argmin(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else (len(x) - 1) - x.argmin()
        
        result = self._apply_rolling(safe_argmin, window)
        
        return self.__class__(result, f"ts_argmin({self.name},{window})")
    
    def ts_argmax(self, window: int) -> Self:
        self._validate_window(window)
        
        def safe_argmax(x: pd.Series) -> float:
            return np.nan if (x.isna().any() or len(x) < window) else (len(x) - 1) - x.argmax()
        
        result = self._apply_rolling(safe_argmax, window)
        
        return self.__class__(result, f"ts_argmax({self.name},{window})")
    
    def ts_scale(self, window: int, constant: float = 0) -> Self:
        self._validate_window(window)
        
        min_factor = self.ts_min(window)
        max_factor = self.ts_max(window)
        
        result = (self - min_factor) / (max_factor - min_factor)
        result._data['factor'] = self._replace_inf(result._data['factor'])
        
        result += constant
        
        return self.__class__(result._data, f"ts_scale({self.name},{window},{constant})")
    
    def ts_zscore(self, window: int) -> Self:
        self._validate_window(window)
        
        mean_factor = self.ts_mean(window)
        std_factor = self.ts_std(window)
        
        result = (self - mean_factor) / std_factor
        result._data['factor'] = self._replace_inf(result._data['factor'])
        
        return self.__class__(result._data, f"ts_zscore({self.name},{window})")
    
    def ts_quantile(self, window: int, driver: str = "gaussian") -> Self:
        self._validate_window(window)
        
        valid_drivers = {
            "gaussian": norm.ppf,
            "uniform": uniform.ppf,
            "cauchy": cauchy.ppf,
        }
        if driver not in valid_drivers:
            raise ValueError(f"Invalid driver: {driver}. Valid drivers are: {list(valid_drivers.keys())}")
        
        ppf_func = valid_drivers[driver]
        ranked_factor = self.ts_rank(window)
        
        result = ranked_factor._data.copy()
        epsilon = 1e-6
        result['factor'] = result['factor'].clip(lower=epsilon, upper=1-epsilon).apply(ppf_func)
        
        
        return self.__class__(result, f"ts_quantile({self.name},{window},{driver})")
    
    def ts_kurtosis(self, window: int) -> Self:
        self._validate_window(window)
        
        result = self._data.copy()
        
        def kurtosis_vectorized(group):
            vals = group.values
            n = len(vals)
            kurt_vals = np.full(n, np.nan)
            
            for i in range(window - 1, n):
                window_vals = vals[i - window+1:i+1]
                
                if np.isnan(window_vals).any():
                    continue
                
                if len(np.unique(window_vals)) < 2:
                    continue
                
                mean_val = np.mean(window_vals)
                std_val = np.std(window_vals, ddof=0)
                
                if std_val < 1e-10:
                    continue
                
                deviations = window_vals - mean_val
                kurt = np.mean(deviations**4) / (std_val**4) - 3
                kurt_vals[i] = kurt
                
            return pd.Series(kurt_vals, index=group.index)
        
        result['factor'] = result.groupby('symbol', group_keys=False)['factor'].apply(
            kurtosis_vectorized
        )
        
        return self.__class__(result, f"ts_kurtosis({self.name},{window})")
    
    def ts_skewness(self, window: int) -> Self:
        self._validate_window(window)
        
        n = window
        mean_val = self.ts_mean(window)
        diff = self - mean_val
        
        sum_cube_dev = diff.pow(3).ts_sum(window)
        sum_square_dev = diff.pow(2).ts_sum(window)
        
        numerator = sum_cube_dev * n
        denominator = sum_square_dev.pow(1.5) * ((n - 1) * (n - 2))
        
        skew = numerator / denominator
        skew.name = f"ts_skewness({self.name},{window})"
        return skew
    
    def ts_step(self, start: int = 1) -> Self:
        result = self._data.copy()
        result['factor'] = result.groupby('symbol').cumcount() + start
        return self.__class__(result, f"ts_step({self.name},{start})")
    
    def ts_shift(self, period: int) -> Self:
        result = self._data.copy()
        result['factor'] = result.groupby('symbol')['factor'].shift(period)
        return self.__class__(result, f"ts_shift({self.name},{period})")
    
    def ts_delta(self, period: int) -> Self:
        result = self._data.copy()
        result['factor'] = result.groupby('symbol')['factor'].diff(period)
        return self.__class__(result, f"ts_delta({self.name},{period})")
    
    def ts_corr(self, other: Self, window: int) -> Self:
        self._validate_window(window)
        self._validate_factor(other, "ts_corr")
        
        merged = pd.merge(self._data, other.data,
                          on=['start_time', 'end_time', 'symbol'],
                          suffixes=('_x', '_y'))
        
        if merged.empty:
            raise ValueError(f"No common data between factors")
        
        def safe_corr(group):
            x = group['factor_x']
            y = group['factor_y']
            
            valid_mask = x.notna() & y.notna()
            if valid_mask.sum() < 2:
                return pd.Series(np.nan, index=group.index)
            
            if x[valid_mask].std() == 0 or y[valid_mask].std() == 0:
                return pd.Series(np.nan, index=group.index)
            
            corr_result = group[['factor_x', 'factor_y']].rolling(
                window, min_periods=window
            ).corr().iloc[0::3, -1]
            
            return corr_result
        
        result = result[['start_time', 'end_time', 'symbol', 'factor']]
        return self.__class__(result, f"ts_corr({self.name},{other.name},{window})")
    
    def ts_cov(self, other: Self, window: int) -> Self:
        self._validate_window(window)
        self._validate_factor(other, "ts_cov")
        
        merged = pd.merge(self.data, other.data,
                          on=['start_time', 'end_time', 'symbol'],
                          suffixes=('_x', '_y'))
        
        if merged.empty:
            raise ValueError(f"No common data between factors")
        
        def safe_cov(group):
            x = group['factor_x']
            y = group['factor_y']
            
            valid_mask = x.notna() & y.notna()
            if valid_mask.sum() < 2:
                return pd.Series(np.nan, index=group.index)
            
            
            cov_result = group[['factor_x', 'factor_y']].rolling(
                window, min_periods=window
            ).cov().iloc[0::3, -1]
            
            return cov_result
        
        result = merged.copy()
        result['factor'] = result.groupby('symbol', group_keys=False).apply(
            safe_cov, include_groups=False
        ).values
        
        result = result[['start_time', 'end_time', 'symbol', 'factor']]
        return self.__class__(result, f"ts_cov({self.name},{other.name},{window})")

    def ts_cv(self, window: int) -> Self:
        """
        Coefficient of Variation
        """
        self._validate_window(window)
        
        result = self._data.copy()
        
        def cv_vectorized(group):
            vals = group.values
            n = len(vals)
            out = np.full(n, np.nan)
            
            for i in range(window - 1, n):
                w = vals[i - window + 1:i + 1]
                
                if np.isnan(w).any() or len(w) < window:
                    continue
                
                mean_val = np.mean(w)
                std_val = np.std(w, ddof=1)
                
                out[i] = std_val / (abs(mean_val) + 1e-10)
            
            return pd.Series(out, index=group.index)
        
        result['factor'] = result.groupby('symbol', group_keys=False)['factor'].apply(cv_vectorized)
        result['factor'] = self._replace_inf(result['factor'])
        return self.__class__(result, f"ts_cv({self.name},{window})")
    
    def ts_jumpiness(self, window: int) -> Self:
        """
        Compares the total path traveled vs the range (max - min).
        """
        self._validate_window(window)
        diff = self.ts_delta(1).abs()
        total_jump = diff.ts_sum(window)
        range_val = self.ts_max(window) - self.ts_min(window)
        result = total_jump / (range_val + 1e-10)
        result.data['factor'] = self._replace_inf(result.data['factor'])
        return self.__class__(result, f"ts_jumpiness({self.name},{window})")
    
    def ts_autocorr(self, window: int, lag: int = 1) -> Self:
        self._validate_window(window)
        if lag <= 0:
            raise ValueError(f"Lag must be positive")
        
        lagged_factor = self.ts_shift(lag)
        result = self.ts_corr(lagged_factor, window)
        return self.__class__(result, f"ts_autocorr({self.name},{window},{lag})")
    
    def ts_reversal_count(self, window: int) -> Self:
        self._validate_window(window)
        
        def count_reversals(s):
            if len(s) < 3:
                return np.nan
            diff = np.diff(s)
            if len(diff) < 2:
                return np.nan
            valid_diff = diff[~np.isnan(diff)]
            if len(valid_diff) < 2:
                return np.nan
            sign_changes = ((valid_diff[1:] * valid_diff[:-1]) < 0).sum()
            return sign_changes / (len(valid_diff) - 1)
        
        result = self._data.copy()
        result['factor'] = (result.groupby('symbol')['factor']
                           .rolling(window, min_periods=3)
                           .apply(count_reversals, raw=True)
                           .reset_index(level=0, drop=True))
        
        return self.__class__(result, f"ts_reversal_count({self.name},{window})")


    def ts_vr(self, window: int, k: int = 2) -> Self:
        """
        Variance Ratio - tests if the market follows a random walk hypothesis.
        
        VR ≈ 1: Random walk
        VR > 1: Trending (positive autocorrelation)
        VR < 1: Mean-reverting (negative autocorrelation)
        """
        self._validate_window(window)
        if k <= 0:
            raise ValueError("k must be positive")
        k_diff = self.ts_delta(k)
        one_diff = self.ts_delta(1)
        var_k = k_diff.ts_std(window) ** 2
        var_1 = one_diff.ts_std(window) ** 2
        result = var_k / (k * var_1 + 1e-10)
        result.data['factor'] = self._replace_inf(result.data['factor'])
        return self.__class__(result, f"ts_vr({self.name},{window},{k})")
