import pandas as pd
import polars as pl
import numpy as np
import logging
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any
from .core import Factor
from ..aggbar import AggBar
import matplotlib.figure as mpl_figure

logger = logging.getLogger(__name__)


@dataclass
class FactorAnalysisResult:
    """
    Structured result from factor analysis.

    Attributes:
        factor_name: Name of the analyzed factor
        periods: Analysis periods (forward return horizons)
        quantiles: Number of quantiles used
        ic_series: Information Coefficient time series
        ic_summary: Summary statistics of IC (mean, std, ir, t-stat)
        quantile_returns: Mean returns by quantile
        cumulative_returns: Cumulative returns by quantile (if available)
    """

    factor_name: str
    periods: int
    quantiles: int
    ic_series: pd.DataFrame
    ic_summary: Dict[str, float]
    quantile_returns: pd.DataFrame
    cumulative_returns: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "factor_name": self.factor_name,
            "periods": self.periods,
            "quantiles": self.quantiles,
            "ic_series": self.ic_series,
            "ic_summary": self.ic_summary,
            "quantile_returns": self.quantile_returns,
            "cumulative_returns": self.cumulative_returns,
        }

    def __repr__(self) -> str:
        ic = self.ic_summary
        return f"""FactorAnalysisResult: {self.factor_name}
  Periods: {self.periods}, Quantiles: {self.quantiles}
  Mean IC: {ic.get("mean_ic", 0):.4f}
  IC Std: {ic.get("ic_std", 0):.4f}
  IC IR: {ic.get("ic_ir", 0):.4f}
"""


class FactorAnalyzer:
    """
    Analyzer for factor performance and characteristics.
    """

    def __init__(self, factor: Factor, prices: Union[AggBar, Factor], quantiles: int = 5):
        self.factor = factor
        self.quantiles = quantiles
        self._raw_prices = prices
        if isinstance(prices, AggBar):
            try:
                self.prices = prices["close"]
            except KeyError:
                # If 'close' is not there, we'll wait for price_col in prepare_data
                self.prices = None
        else:
            self.prices = prices

    def analyze(self, price_col: str = "close", periods: int = 1) -> FactorAnalysisResult:
        """
        Run full factor analysis.

        Returns:
            FactorAnalysisResult with IC series, summary, and quantile returns
        """
        # Prepare data
        self.prepare_data(price_col=price_col, periods=[periods])

        # Calculate IC
        ic_series = self.calculate_ic()
        ic_summary_df = self.calculate_ic_summary()

        # Convert IC summary to dict for single period as expected by FactorAnalysisResult
        col = f"period_{periods}"
        ic_summary = {
            "mean_ic": ic_summary_df.loc["mean", col] if col in ic_summary_df.columns else 0.0,
            "ic_std": ic_summary_df.loc["std", col] if col in ic_summary_df.columns else 0.0,
            "ic_ir": ic_summary_df.loc["ic_ir", col] if col in ic_summary_df.columns else 0.0,
            "t-stat": ic_summary_df.loc["t-stat", col] if col in ic_summary_df.columns else 0.0,
        }

        # Calculate quantile returns
        quantile_returns = self.calculate_quantile_returns(quantiles=self.quantiles, period=periods)

        # Calculate cumulative returns (optional)
        try:
            cumulative_returns = self.calculate_cumulative_returns(quantiles=self.quantiles, period=periods)
        except Exception:
            cumulative_returns = None

        return FactorAnalysisResult(
            factor_name=self.factor.name,
            periods=periods,
            quantiles=self.quantiles,
            ic_series=ic_series,
            ic_summary=ic_summary,
            quantile_returns=quantile_returns,
            cumulative_returns=cumulative_returns,
        )

    def prepare_data(self, periods: Optional[List[int]] = None, price_col: Optional[str] = None) -> pl.DataFrame:
        """
        Prepare data for analysis by aligning factor values with future returns.

        Args:
            periods: List of holding periods to calculate future returns for.
            price_col: Column name for prices if prices was provided as AggBar.

        Returns:
            pl.DataFrame: Merged data with 'factor' and 'period_n' returns.
        """
        if periods is None:
            periods = [1, 5, 10]

        # Get factor data
        factor_lf = self.factor.lazy
        # Trigger a lightweight count to check if empty
        if factor_lf.select(pl.len()).collect().item() == 0:
            raise ValueError("Factor data is empty.")

        # Get price data
        if price_col is not None and isinstance(self._raw_prices, AggBar):
            prices_lf = self._raw_prices.to_polars().lazy().select(["start_time", "end_time", "symbol", price_col])
            price_col_name = price_col
        elif self.prices is not None:
            # self.prices is a Factor
            prices_lf = self.prices.lazy.rename({"factor": "__price__"})
            price_col_name = "__price__"
        else:
            raise ValueError("No price data available. Provide price_col or initialize with prices.")

        # Align and merge using Polars
        # Use inner join to ensure we have both factor and prices
        df_lf = factor_lf.join(
            prices_lf,
            on=["start_time", "end_time", "symbol"],
            how="inner",
        )

        # Calculate forward returns for each period
        # return = (price.shift(-p) / price) - 1.0
        return_exprs = []
        for p in periods:
            return_exprs.append(
                ((pl.col(price_col_name).shift(-p).over("symbol") / pl.col(price_col_name)) - 1.0).alias(f"period_{p}")
            )

        df_lf = df_lf.with_columns(return_exprs)

        # Drop any remaining NaNs to ensure strict data alignment
        self._clean_data = df_lf.collect().drop_nulls()

        original_count = factor_lf.select(pl.len()).collect().item()
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
        corr_method = "spearman" if method == "rank" else "pearson"

        ic_df = (
            self._clean_data.group_by("start_time")
            .agg([pl.corr("factor", col, method=corr_method).alias(col) for col in period_cols])
            .sort("start_time")
        )

        return ic_df.to_pandas().set_index("start_time")

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
            count = len(vals)
            t_stat = mean / (std / np.sqrt(count)) if std > 0 and count > 0 else np.nan
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

        # Assign quantiles using Polars rank-based approach
        df = self._clean_data.with_columns(pl.col("factor").rank(method="random").over("start_time").alias("_rank"))

        df = df.with_columns(
            ((pl.col("_rank") - 1) / pl.len().over("start_time") * quantiles)
            .floor()
            .cast(pl.Int32)
            .add(1)
            .alias("quantile")
        )

        # Group by time and quantile
        q_ret = (
            df.group_by(["start_time", "quantile"])
            .agg([pl.col(col).mean().alias("mean_ret"), pl.len().alias("count")])
            .sort(["start_time", "quantile"])
        )

        return q_ret.to_pandas().set_index(["start_time", "quantile"])

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
