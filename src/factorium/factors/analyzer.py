import pandas as pd
import numpy as np
import logging
from typing import Union, List, Optional
from .core import Factor
from ..aggbar import AggBar
import matplotlib.figure as mpl_figure

logger = logging.getLogger(__name__)


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
        original_count = len(self.factor.data)
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
        self._clean_data = df.dropna()
        final_count = len(self._clean_data)
        retained_pct = (final_count / original_count * 100) if original_count > 0 else 0
        logger.info(f"prepare_data: {original_count} rows -> {final_count} rows ({retained_pct:.1f}% retained)")
        return self._clean_data

    def calculate_ic(self, method: str = "rank") -> pd.DataFrame:
        """
        Calculate Information Coefficient (IC) for each period.

        Args:
            method: 'rank' for Spearman rank correlation, 'normal' for Pearson correlation.

        Returns:
            pd.DataFrame: IC values indexed by start_time.
        """
        if not hasattr(self, "_clean_data"):
            raise ValueError("Data not prepared. Call prepare_data() first.")

        period_cols = [c for c in self._clean_data.columns if c.startswith("period_")]

        def _group_ic(group):
            if len(group) < 2:
                return pd.Series({c: np.nan for c in period_cols}, dtype=float)

            res = {}
            corr_method = "spearman" if method == "rank" else "pearson"
            for c in period_cols:
                res[c] = group["factor"].corr(group[c], method=corr_method)
            return pd.Series(res)

        ic = self._clean_data.groupby("start_time").apply(_group_ic, include_groups=False)
        return ic

    def calculate_ic_summary(self, method: str = "rank") -> pd.DataFrame:
        """
        Calculate summary statistics for IC.

        Returns:
            pd.DataFrame: Summary statistics (mean, std, t-stat, ic_ir).
        """
        ic = self.calculate_ic(method=method)
        summary = {}

        for col in ic.columns:
            vals = ic[col].dropna()
            if vals.empty:
                summary[col] = {"mean": np.nan, "std": np.nan, "t-stat": np.nan, "ic_ir": np.nan}
                continue

            mean = vals.mean()
            std = vals.std()
            t_stat = mean / (std / np.sqrt(len(vals))) if std > 0 else np.nan
            ic_ir = mean / std if std > 0 else np.nan

            summary[col] = {
                "mean": mean,
                "std": std,
                "t-stat": t_stat,
                "ic_ir": ic_ir,
            }

        return pd.DataFrame(summary)

    def calculate_quantile_returns(self, quantiles: int = 5, period: int = 1) -> pd.DataFrame:
        """
        Calculate mean returns for each factor quantile.

        Args:
            quantiles: Number of quantiles to split the factor into.
            period: The return period to use.

        Returns:
            pd.DataFrame: Mean returns and counts per (start_time, quantile).
        """
        if not hasattr(self, "_clean_data"):
            raise ValueError("Data not prepared. Call prepare_data() first.")

        col = f"period_{period}"
        if col not in self._clean_data.columns:
            raise ValueError(f"Return for period {period} not found in prepared data.")

        df = self._clean_data.copy()

        def assign_quantiles(x):
            try:
                return pd.qcut(x, quantiles, labels=False, duplicates="drop") + 1
            except ValueError:
                return pd.Series([np.nan] * len(x), index=x.index)

        df["quantile"] = df.groupby("start_time", group_keys=False)["factor"].apply(assign_quantiles)
        df = df.dropna(subset=["quantile"])

        # Group by time and quantile
        q_ret = df.groupby(["start_time", "quantile"])[col].agg(["mean", "count"]).rename(columns={"mean": "mean_ret"})
        return q_ret

    def calculate_cumulative_returns(
        self, quantiles: int = 5, period: int = 1, long_short: bool = True
    ) -> pd.DataFrame:
        """
        Calculate cumulative returns for each factor quantile.

        Args:
            quantiles: Number of quantiles.
            period: The return period to use.
            long_short: Whether to include a Long-Short (Top - Bottom) portfolio.

        Returns:
            pd.DataFrame: Cumulative returns indexed by start_time.
        """
        q_ret = self.calculate_quantile_returns(quantiles=quantiles, period=period)

        # Pivot to have quantiles as columns
        q_ret_pivot = q_ret["mean_ret"].unstack("quantile")

        if long_short and not q_ret_pivot.empty:
            top_q = q_ret_pivot.columns.max()
            bottom_q = q_ret_pivot.columns.min()
            if top_q != bottom_q:
                q_ret_pivot["Long-Short"] = q_ret_pivot[top_q] - q_ret_pivot[bottom_q]

        # Cumulative returns: (1 + r).cumprod() - 1
        cum_ret = (1 + q_ret_pivot).cumprod() - 1
        return cum_ret

    def plot_ic(self, period: int = 1, method: str = "rank", plot_type: str = "ts") -> mpl_figure.Figure:
        """
        Plot Information Coefficient (IC).

        Args:
            period: The return period to use.
            method: 'rank' or 'normal'.
            plot_type: 'ts' for time series, 'hist' for histogram.
        """
        from .plotting_analyzer import FactorAnalyzerPlotter

        ic = self.calculate_ic(method=method)
        col = f"period_{period}"
        if col not in ic.columns:
            raise ValueError(f"Period {period} not found in IC data.")

        plotter = FactorAnalyzerPlotter()
        if plot_type == "ts":
            return plotter.plot_ic_ts(ic[[col]])
        elif plot_type == "hist":
            return plotter.plot_ic_hist(ic[[col]])
        else:
            raise ValueError(f"Invalid plot_type: {plot_type}. Expected 'ts' or 'hist'.")

    def plot_quantile_returns(self, quantiles: int = 5, period: int = 1) -> mpl_figure.Figure:
        """
        Plot mean returns for each factor quantile.
        """
        from .plotting_analyzer import FactorAnalyzerPlotter

        q_ret = self.calculate_quantile_returns(quantiles=quantiles, period=period)
        plotter = FactorAnalyzerPlotter()
        return plotter.plot_quantile_returns(q_ret)

    def plot_cumulative_returns(
        self, quantiles: int = 5, period: int = 1, long_short: bool = True
    ) -> mpl_figure.Figure:
        """
        Plot cumulative returns for each factor quantile.
        """
        from .plotting_analyzer import FactorAnalyzerPlotter

        cum_ret = self.calculate_cumulative_returns(quantiles=quantiles, period=period, long_short=long_short)
        plotter = FactorAnalyzerPlotter()
        return plotter.plot_cumulative_returns(cum_ret)
