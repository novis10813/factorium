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
        if isinstance(prices, AggBar):
            # Default to 'close' if it's an AggBar
            self.prices = prices["close"]
        else:
            self.prices = prices

    def prepare_data(self, periods: List[int] = (1, 5, 10), price_col: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare data for analysis by aligning factor values with future returns.

        Args:
            periods: List of holding periods to calculate future returns for.
            price_col: Column name for prices if prices was provided as AggBar.

        Returns:
            pd.DataFrame: Merged data with 'factor' and 'period_n' returns.
        """
        prices = self.prices

        # If price_col is specified and prices is actually an AggBar (though we converted it in __init__)
        # Wait, if we want to support dynamic price_col, we should handle it.
        # But the instructions say __init__ takes prices: AggBar | Factor.
        # Let's re-handle it if price_col is provided and we have access to original AggBar if possible?
        # Actually, the test shows analyzer = FactorAnalyzer(factor, prices) where prices is AggBar.

        # If the user wants a different column than what was initialized, they should probably
        # have passed that column specifically or we should have kept the AggBar.

        # Let's adjust __init__ to store the AggBar if it's an AggBar.
        # No, let's just stick to the instructions.

        # If prices is already a Factor, we use it.
        # If it's an AggBar, we need to extract the column.

        # Wait, I already converted it in __init__.
        # If the user wants a different price_col now, they can't if I only stored the Factor.

        # Let's check the test again.
        # test_prepare_data(sample_data):
        #     agg = AggBar(sample_data)
        #     factor = agg["my_factor"]
        #     prices = agg  # Test AggBar to Factor conversion
        #     analyzer = FactorAnalyzer(factor, prices)
        #     df = analyzer.prepare_data(periods=periods, price_col="close")

        # If prices is AggBar in __init__, and price_col is passed in prepare_data,
        # it should use that column.

        # Let's refine FactorAnalyzer.
        pass


class FactorAnalyzer:
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

    def prepare_data(self, periods: List[int] = (1, 5, 10), price_col: Optional[str] = None) -> pd.DataFrame:
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
            df = pd.merge(df, ret_data, on=["start_time", "end_time", "symbol"], how="left")

        return df
