import logging
from dataclasses import dataclass
from typing import Any

import matplotlib.figure as mpl_figure
import numpy as np
import pandas as pd
import polars as pl

from ..aggbar import AggBar
from .core import Factor

logger = logging.getLogger(__name__)


@dataclass
class FactorAnalysisResult:
    """
    Structured result from factor analysis.

    Attributes:
        factor_name: Name of the analyzed factor
        periods: Analysis periods (forward return horizons) - always a list
        quantiles: Number of quantiles used
        ic_series: Information Coefficient time series
        ic_summary: Summary statistics of IC, keyed by period
            Dict[int, Dict[str, float]] with mean_ic, ic_std, ic_ir, t-stat
        turnover_series: Turnover time series (1 - rank autocorrelation)
        turnover_mean: Average turnover across all periods
        quantile_returns: Mean returns by quantile, keyed by period
            Dict[int, pd.DataFrame]
        cumulative_returns: Cumulative returns by quantile (if available)
            Dict[int, pd.DataFrame] or None
    """

    factor_name: str
    periods: list[int]
    quantiles: int
    ic_series: pd.DataFrame
    ic_summary: dict[int, dict[str, float]]
    turnover_series: pd.Series
    turnover_mean: float
    quantile_returns: dict[int, pd.DataFrame]
    cumulative_returns: dict[int, pd.DataFrame] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "factor_name": self.factor_name,
            "periods": self.periods,
            "quantiles": self.quantiles,
            "ic_series": self.ic_series,
            "ic_summary": self.ic_summary,
            "turnover_series": self.turnover_series,
            "turnover_mean": self.turnover_mean,
            "quantile_returns": self.quantile_returns,
            "cumulative_returns": self.cumulative_returns,
        }

    def __repr__(self) -> str:
        lines = [f"FactorAnalysisResult: {self.factor_name}"]
        lines.append(f"  Periods: {self.periods}, Quantiles: {self.quantiles}")
        for p in self.periods:
            ic = self.ic_summary.get(p, {})
            lines.append(f"  Period {p}: IC={ic.get('mean_ic', 0):.4f}, IR={ic.get('ic_ir', 0):.4f}")
        lines.append(f"  Turnover: {self.turnover_mean:.4f}")
        return "\n".join(lines) + "\n"

    def save(self, output_dir: str) -> None:
        """
        Save analysis results to directory with timestamp.

        Creates structure (single horizon):
        {output_dir}/
        └── YYYYMMDD_HHMMSS_{factor_name}/
            ├── config.json
            ├── ic_series.csv
            ├── ic_summary.csv
            ├── turnover.csv
            ├── quantile_returns.csv
            ├── cumulative_returns.csv
            └── plots/
                ├── ic_distribution.png
                ├── ic_timeseries.png
                ├── quantile_returns.png
                └── cumulative_returns.png

        Multi-horizon structure (periods=[1, 5, 20]):
        {output_dir}/
        └── YYYYMMDD_HHMMSS_{factor_name}/
            ├── config.json
            ├── ic_series.csv                   # columns: period_1, period_5, period_20
            ├── ic_summary.csv                  # rows indexed by period
            ├── turnover.csv
            ├── quantile_returns_period_1.csv   # per-period files
            ├── quantile_returns_period_5.csv
            ├── quantile_returns_period_20.csv
            ├── cumulative_returns_period_1.csv
            ├── cumulative_returns_period_5.csv
            ├── cumulative_returns_period_20.csv
            └── plots/
                ├── ic_distribution.png
                ├── ic_timeseries.png
                ├── ic_decay.png                # IC decay curve (multi-horizon only)
                ├── quantile_returns_period_1.png
                ├── quantile_returns_period_5.png
                ├── quantile_returns_period_20.png
                ├── cumulative_returns_period_1.png
                ├── cumulative_returns_period_5.png
                └── cumulative_returns_period_20.png

        Args:
            output_dir: Base directory for experiment outputs
        """
        import json
        from datetime import datetime
        from pathlib import Path

        import matplotlib.pyplot as plt

        from .plotting_analyzer import FactorAnalyzerPlotter

        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{self.factor_name}"
        exp_path = Path(output_dir) / folder_name
        exp_path.mkdir(parents=True, exist_ok=True)

        # Create plots subdirectory
        plots_path = exp_path / "plots"
        plots_path.mkdir(exist_ok=True)

        # Save CSV files
        self.ic_series.to_csv(exp_path / "ic_series.csv")

        # Convert ic_summary to DataFrame for CSV (always dict[int, dict[str, float]] now)
        ic_summary_df = pd.DataFrame(self.ic_summary).T
        ic_summary_df.index.name = "period"
        ic_summary_df.to_csv(exp_path / "ic_summary.csv")

        self.turnover_series.to_csv(exp_path / "turnover.csv", header=True)

        # Handle quantile_returns (always dict[int, pd.DataFrame] now)
        for p, df in self.quantile_returns.items():
            df.to_csv(exp_path / f"quantile_returns_period_{p}.csv")

        if self.cumulative_returns is not None:
            # Always dict[int, pd.DataFrame] now
            for p, df in self.cumulative_returns.items():
                df.to_csv(exp_path / f"cumulative_returns_period_{p}.csv")

        # Save plots
        plotter = FactorAnalyzerPlotter()

        # IC time series plot
        try:
            fig_ic_ts = plotter.plot_ic_ts(self.ic_series)
            fig_ic_ts.savefig(plots_path / "ic_timeseries.png", dpi=150, bbox_inches="tight")
            plt.close(fig_ic_ts)
        except Exception as e:
            logger.warning(f"Failed to generate IC timeseries plot: {e}")

        # IC distribution plot
        try:
            fig_ic_hist = plotter.plot_ic_hist(self.ic_series)
            fig_ic_hist.savefig(plots_path / "ic_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig_ic_hist)
        except Exception as e:
            logger.warning(f"Failed to generate IC distribution plot: {e}")

        # Quantile returns plot (always per-period now)
        for p, df in self.quantile_returns.items():
            try:
                fig_qret = plotter.plot_quantile_returns(df)
                fig_qret.savefig(plots_path / f"quantile_returns_period_{p}.png", dpi=150, bbox_inches="tight")
                plt.close(fig_qret)
            except Exception as e:
                logger.warning(f"Failed to generate quantile returns plot for period {p}: {e}")

        # Cumulative returns plot (if available, always per-period now)
        if self.cumulative_returns is not None:
            for p, df in self.cumulative_returns.items():
                try:
                    fig_cumret = plotter.plot_cumulative_returns(df)
                    fig_cumret.savefig(plots_path / f"cumulative_returns_period_{p}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig_cumret)
                except Exception as e:
                    logger.warning(f"Failed to generate cumulative returns plot for period {p}: {e}")

        # IC decay plot (multi-horizon only)
        if isinstance(self.periods, list) and len(self.periods) > 1:
            try:
                fig_decay = plotter.plot_ic_decay(self.ic_summary)
                fig_decay.savefig(plots_path / "ic_decay.png", dpi=150, bbox_inches="tight")
                plt.close(fig_decay)
            except Exception as e:
                logger.warning(f"Failed to generate IC decay plot: {e}")

        # Save config.json
        config = {
            "factor_name": self.factor_name,
            "periods": self.periods,
            "quantiles": self.quantiles,
            "created_at": datetime.now().isoformat(),
            "data_range": {
                "start": str(self.ic_series.index.min()),
                "end": str(self.ic_series.index.max()),
                "n_observations": len(self.ic_series),
            },
        }

        with open(exp_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Results saved to {exp_path}")


class FactorAnalyzer:
    """
    Analyzer for factor performance and characteristics.
    """

    prices: Factor | None  # Type annotation for prices attribute

    def __init__(self, factor: Factor, prices: AggBar | Factor, quantiles: int = 5):
        self.factor = factor
        self.quantiles = quantiles
        self._raw_prices = prices
        if isinstance(prices, AggBar):
            try:
                close_col = prices["close"]
                if isinstance(close_col, Factor):
                    self.prices = close_col
                else:
                    # close column is not a Factor (AggBar), skip
                    self.prices = None
            except KeyError:
                # If 'close' is not there, we'll wait for price_col in prepare_data
                self.prices = None
        else:
            self.prices = prices

    def _ensure_data_prepared(self, periods: list[int] | None = None, price_col: str | None = None) -> None:
        """Ensure data is prepared. Auto-calls prepare_data() if needed."""
        if not hasattr(self, "_clean_data"):
            logger.info("Data not prepared. Auto-calling prepare_data()...")
            self.prepare_data(periods=periods, price_col=price_col)

    def analyze(self, price_col: str = "close", periods: int | list[int] = 1) -> FactorAnalysisResult:
        """
        Run full factor analysis.

        Args:
            price_col: Column name for prices.
            periods: Single period (int) or list of periods for multi-horizon analysis.

        Returns:
            FactorAnalysisResult with IC series, summary, and quantile returns

        Raises:
            ValueError: If periods is an empty list.
        """
        # Normalize periods to list for internal processing
        periods_list = [periods] if isinstance(periods, int) else periods

        # Validate periods
        if not periods_list:
            raise ValueError("Periods list cannot be empty.")

        # Prepare data
        self.prepare_data(price_col=price_col, periods=periods_list)

        # Calculate IC
        ic_series = self.calculate_ic()
        ic_summary_df = self.calculate_ic_summary()

        # Build ic_summary - always use dict[int, dict[str, float]] format
        ic_summary: dict[int, dict[str, float]] = {}
        for p in periods_list:
            col = f"period_{p}"
            ic_summary[p] = {
                "mean_ic": float(ic_summary_df.loc["mean", col]) if col in ic_summary_df.columns else 0.0,
                "ic_std": float(ic_summary_df.loc["std", col]) if col in ic_summary_df.columns else 0.0,
                "ic_ir": float(ic_summary_df.loc["ic_ir", col]) if col in ic_summary_df.columns else 0.0,
                "t-stat": float(ic_summary_df.loc["t-stat", col]) if col in ic_summary_df.columns else 0.0,
            }

        # Calculate quantile returns - always use dict[int, pd.DataFrame] format
        quantile_returns: dict[int, pd.DataFrame] = {
            p: self.calculate_quantile_returns(quantiles=self.quantiles, period=p) for p in periods_list
        }

        # Calculate cumulative returns (optional) - always use dict[int, pd.DataFrame] format
        try:
            cumulative_returns: dict[int, pd.DataFrame] | None = {
                p: self.calculate_cumulative_returns(quantiles=self.quantiles, period=p) for p in periods_list
            }
        except Exception:
            cumulative_returns = None

        # Calculate turnover
        turnover_series = self.calculate_turnover()
        turnover_mean = float(turnover_series.mean())

        return FactorAnalysisResult(
            factor_name=self.factor.name,
            periods=periods_list,
            quantiles=self.quantiles,
            ic_series=ic_series,
            ic_summary=ic_summary,
            turnover_series=turnover_series,
            turnover_mean=turnover_mean,
            quantile_returns=quantile_returns,
            cumulative_returns=cumulative_returns,
        )

    def prepare_data(self, periods: list[int] | None = None, price_col: str | None = None) -> pl.DataFrame:
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
            # Check if price_col exists in AggBar
            if price_col not in self._raw_prices._data.columns:
                available_cols = ", ".join(sorted(self._raw_prices._data.columns))
                raise ValueError(f"Price column '{price_col}' not found in AggBar. Available columns: {available_cols}")
            prices_lf = self._raw_prices.to_polars().lazy().select(["start_time", "end_time", "symbol", price_col])
            price_col_name = price_col
        elif self.prices is not None:
            # self.prices is a Factor (we've narrowed the type above)
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
        self._ensure_data_prepared()

        period_cols = [c for c in self._clean_data.columns if c.startswith("period_")]
        corr_method = "spearman" if method == "rank" else "pearson"

        ic_df = (
            self._clean_data.group_by("start_time")
            .agg([pl.corr("factor", col, method=corr_method).alias(col) for col in period_cols])  # type: ignore[call-overload]
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

    def calculate_turnover(self) -> pd.Series:
        """
        Calculate factor turnover using rank autocorrelation.

        Method:
        1. For each start_time, calculate cross-sectional rank of factor values
        2. Compute correlation between today's rank and yesterday's rank
        3. turnover = 1 - rank_autocorrelation

        Returns:
            pd.Series: Turnover time series indexed by start_time
        """
        self._ensure_data_prepared()

        # Ensure data is sorted by symbol and time for correct shift operation
        sorted_data = self._clean_data.sort(["symbol", "start_time"])

        # Calculate rank per start_time and get previous rank for each symbol
        turnover_df = (
            sorted_data.with_columns(pl.col("factor").rank(method="average").over("start_time").alias("factor_rank"))
            .with_columns(
                # Get previous period's rank for each symbol
                pl.col("factor_rank").shift(1).over("symbol").alias("prev_rank")
            )
            .group_by("start_time")
            .agg(
                # Calculate correlation between current rank and previous rank
                pl.corr("factor_rank", "prev_rank").alias("autocorr")
            )
            .select([pl.col("start_time"), (1 - pl.col("autocorr")).alias("turnover")])
            .sort("start_time")
        )

        # Convert to pandas Series only at the end
        return turnover_df.to_pandas().set_index("start_time")["turnover"]

    def calculate_quantile_returns(self, quantiles: int = 5, period: int = 1) -> pd.DataFrame:
        """
        Calculate mean returns for each factor quantile.

        Args:
            quantiles: Number of quantiles to split the factor into.
            period: The return period to use.

        Returns:
            pd.DataFrame: Mean returns and counts per (start_time, quantile).
        """
        self._ensure_data_prepared(periods=[period])

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

    def plot_ic_decay(self, periods: list[int] | None = None, method: str = "rank") -> mpl_figure.Figure:
        """
        Plot IC decay curve across multiple horizons.

        Args:
            periods: List of periods to plot. If None, uses all available periods.
            method: 'rank' for Spearman, 'normal' for Pearson.

        Returns:
            matplotlib Figure
        """
        from .plotting_analyzer import FactorAnalyzerPlotter

        ic_summary_df = self.calculate_ic_summary(method=method)

        # Build ic_summary dict for plotting
        if periods is None:
            # Extract periods from available columns
            periods = [int(c.replace("period_", "")) for c in ic_summary_df.columns if c.startswith("period_")]

        ic_summary = {}
        for p in periods:
            col = f"period_{p}"
            if col in ic_summary_df.columns:
                ic_summary[p] = {
                    "mean_ic": float(ic_summary_df.loc["mean", col]),
                    "ic_ir": float(ic_summary_df.loc["ic_ir", col]),
                }

        if not ic_summary:
            raise ValueError("No IC data available for the specified periods.")

        plotter = FactorAnalyzerPlotter()
        return plotter.plot_ic_decay(ic_summary)
