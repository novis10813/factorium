"""
Plotting utilities for FactorAnalyzer.
"""


import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import pandas as pd


class FactorAnalyzerPlotter:
    """
    Plotting utility for FactorAnalyzer results.
    """

    def plot_ic_ts(self, ic_data: pd.DataFrame, figsize: tuple[int, int] = (12, 6)) -> mpl_figure.Figure:
        """
        Plot time series of IC.

        Args:
            ic_data: IC values indexed by time.
            figsize: Figure size.

        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ic_data.plot(ax=ax)
        ax.set_title("Information Coefficient (IC) Time Series")
        ax.set_xlabel("Time")
        ax.set_ylabel("IC")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_ic_hist(self, ic_data: pd.DataFrame, figsize: tuple[int, int] = (10, 6)) -> mpl_figure.Figure:
        """
        Plot histogram of IC.

        Args:
            ic_data: IC values.
            figsize: Figure size.

        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ic_data.plot(kind="hist", bins=50, alpha=0.7, ax=ax, edgecolor="black")
        ax.set_title("Information Coefficient (IC) Distribution")
        ax.set_xlabel("IC")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_quantile_returns(
        self, quantile_stats: pd.DataFrame, figsize: tuple[int, int] = (10, 6)
    ) -> mpl_figure.Figure:
        """
        Plot bar chart of mean returns per quantile.

        Args:
            quantile_stats: Mean returns per (time, quantile).
            figsize: Figure size.

        Returns:
            matplotlib Figure object.
        """
        # quantile_stats is indexed by (start_time, quantile)
        # We want the average of mean_ret over all start_time per quantile
        mean_returns = quantile_stats.groupby("quantile")["mean_ret"].mean()

        fig, ax = plt.subplots(figsize=figsize)
        mean_returns.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
        ax.set_title("Mean Returns by Factor Quantile")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Mean Return")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_cumulative_returns(self, cum_ret: pd.DataFrame, figsize: tuple[int, int] = (12, 6)) -> mpl_figure.Figure:
        """
        Plot cumulative returns of quantiles.

        Args:
            cum_ret: Cumulative returns indexed by time.
            figsize: Figure size.

        Returns:
            matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        cum_ret.plot(ax=ax)
        ax.set_title("Cumulative Returns by Factor Quantile")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_ic_decay(self, ic_summary: dict[int, dict[str, float]]) -> mpl_figure.Figure:
        """
        Plot IC decay curve across horizons.

        Args:
            ic_summary: Dict mapping period -> {"mean_ic": float, "ic_ir": float, ...}

        Returns:
            matplotlib Figure
        """
        periods = sorted(ic_summary.keys())
        mean_ics = [ic_summary[p]["mean_ic"] for p in periods]
        ic_irs = [ic_summary[p]["ic_ir"] for p in periods]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Mean IC decay
        ax1.bar(range(len(periods)), mean_ics, color="steelblue", alpha=0.7)
        ax1.set_xticks(range(len(periods)))
        ax1.set_xticklabels([str(p) for p in periods])
        ax1.set_xlabel("Horizon (periods)")
        ax1.set_ylabel("Mean IC")
        ax1.set_title("IC Decay by Horizon")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # IC IR decay
        ax2.bar(range(len(periods)), ic_irs, color="darkorange", alpha=0.7)
        ax2.set_xticks(range(len(periods)))
        ax2.set_xticklabels([str(p) for p in periods])
        ax2.set_xlabel("Horizon (periods)")
        ax2.set_ylabel("IC IR")
        ax2.set_title("IC Information Ratio by Horizon")
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        fig.tight_layout()
        return fig
