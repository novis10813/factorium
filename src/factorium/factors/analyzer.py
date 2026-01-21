import pandas as pd
import numpy as np
from typing import Union, List, Optional
from .core import Factor
from ..aggbar import AggBar


class FactorAnalyzer:
    """
    Analyzer for factor performance and characteristics.
    """

    def __init__(self, factor: Factor, prices: Union[AggBar, Factor]):
        self.factor = factor
        self._raw_prices = prices
        if isinstance(prices, AggBar):
            try:
                self.prices = prices["close"]
            except KeyError:
                # If 'close' is not there, we'll wait for price_col in prepare_data
                self.prices = None
        else:
            self.prices = prices

    def prepare_data(self, periods: Optional[List[int]] = None, price_col: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare data for analysis by aligning factor values with future returns.

        Args:
            periods: List of holding periods to calculate future returns for.
            price_col: Column name for prices if prices was provided as AggBar.

        Returns:
            pd.DataFrame: Merged data with 'factor' and 'period_n' returns.
        """
        if self.factor.data.empty:
            raise ValueError("Factor data is empty.")

        if periods is None:
            periods = [1, 5, 10]
        if price_col is not None and isinstance(self._raw_prices, AggBar):
            prices_factor = self._raw_prices[price_col]
        elif self.prices is not None:
            prices_factor = self.prices
        else:
            raise ValueError("No price data available. Provide price_col or initialize with prices.")

        # Calculate returns: (prices.ts_shift(-p) - prices) / prices
        returns = {}
        for p in periods:
            # Future price: prices shifted by -p
            future_price = prices_factor.ts_shift(-p)
            ret = (future_price - prices_factor) / prices_factor
            returns[f"period_{p}"] = ret

        # Align and merge
        # Start with factor data
        df = self.factor.data.copy()

        for name, ret_factor in returns.items():
            ret_data = ret_factor.data.rename(columns={"factor": name})
            # Use inner join to ensure we have both factor and returns
            df = pd.merge(df, ret_data, on=["start_time", "end_time", "symbol"], how="inner")

        # Drop any remaining NaNs to ensure strict data alignment
        return df.dropna()
