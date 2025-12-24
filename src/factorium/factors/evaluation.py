import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from .core import Factor

class FactorEvaluator:
    """
    Evaluates the performance of a Factor signal.
    """
    
    def __init__(self, factor: "Factor", prices: "Factor"):
        self.factor = factor
        self.prices = prices
        self._results = {}
        
    def _prepare_data(self, periods: List[int]) -> pd.DataFrame:
        """
        Prepare combined dataframe with factor values and forward returns.
        """
        # Convert factor and price to pivoted format for easier calculations
        signal_df = self.factor.data.pivot(index='end_time', columns='symbol', values='factor')
        price_df = self.prices.data.pivot(index='end_time', columns='symbol', values='factor')
        
        # Calculate forward returns for each period
        combined_data = []
        
        for period in periods:
            # Forward returns: (P_{t+period} / P_t) - 1
            # We use shift(-period) to align future return with current signal
            fwd_returns = price_df.pct_change(period).shift(-period)
            
            # Melt back to long format
            melted_returns = fwd_returns.reset_index().melt(id_vars='end_time', value_name=f'return_{period}d')
            combined_data.append(melted_returns)
            
        # Merge signal
        signal_melted = self.factor.data[['end_time', 'symbol', 'factor']]
        
        final_df = signal_melted.copy()
        for df in combined_data:
            final_df = pd.merge(final_df, df, on=['end_time', 'symbol'], how='left')
            
        return final_df

    def calculate_ic(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Calculate Rank IC (Information Coefficient) for each period.
        """
        ic_results = []
        
        for period in periods:
            col_name = f'return_{period}d'
            # Group by time and calculate spearman correlation between factor and returns
            daily_ic = data.groupby('end_time').apply(
                lambda x: x['factor'].corr(x[col_name], method='spearman'),
                include_groups=False
            )
            ic_results.append(pd.Series(daily_ic, name=f'IC_{period}d'))
            
        return pd.concat(ic_results, axis=1)

    def calculate_turnover(self, factor_data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor turnover rate (rank autocorrelation).
        """
        # Pivot factor values
        pivoted = self.factor.data.pivot(index='end_time', columns='symbol', values='factor')
        # Calculate rank autocorrelation day by day
        ranks = pivoted.rank(axis=1, pct=True)
        turnover = ranks.corrwith(ranks.shift(1), axis=1)
        return turnover

    def run_layer_test(self, data: pd.DataFrame, periods: List[int], quantiles: int = 5) -> Dict[int, pd.DataFrame]:
        """
        Calculate average returns for each quantile group using global quantiles
        over the entire sample (not per-day cross-sectional buckets).
        """
        layer_results = {}
        
        data = data.copy()
        
        # Assign global quantiles based on the full sample of factor values.
        # This groups all (time, symbol) observations into quantiles according
        # to their factor scores, which is more aligned with analyzing how
        # factor level relates to future returns across the whole period.
        valid_mask = data['factor'].notna()
        if valid_mask.any():
            data.loc[valid_mask, 'quantile'] = pd.qcut(
                data.loc[valid_mask, 'factor'],
                quantiles,
                labels=False,
                duplicates='drop'
            )
        else:
            data['quantile'] = np.nan
        
        for period in periods:
            col_name = f'return_{period}d'
            # Group by quantile and calculate mean return
            layer_returns = data.groupby('quantile')[col_name].mean()
            layer_results[period] = layer_returns
            
        return layer_results

    def run_full_report(self, periods: List[int] = (1, 5, 10), quantiles: int = 5, save_path: str = None, **kwargs) -> Dict[str, Any]:
        """
        Run a full evaluation report and return summary statistics.
        """
        data = self._prepare_data(periods)
        
        # 1. IC analysis
        ic_series = self.calculate_ic(data, periods)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std
        
        # 2. Turnover analysis
        turnover = self.calculate_turnover(data)
        
        # 3. Layer test
        layer_returns = self.run_layer_test(data, periods, quantiles)
        
        # 4. Long-Short (Spread) Calculation
        spread_results = {}
        for period in periods:
            # Top quantile - Bottom quantile
            spread = layer_returns[period].iloc[-1] - layer_returns[period].iloc[0]
            spread_results[period] = spread

        results = {
            'ic_mean': ic_mean,
            'ic_ir': ic_ir,
            'ic_series': ic_series,
            'turnover_mean': turnover.mean(),
            'turnover_series': turnover,
            'layer_returns': layer_returns,
            'spread': spread_results
        }
        
        if save_path:
            self.plot_report(results, save_path)
            
        return results

    def plot_report(self, results: Dict[str, Any], save_path: str):
        """
        Generate and save a visualization report of the factor evaluation.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Factor Evaluation Report: {self.factor.name}", fontsize=16)
        
        # 1. IC Time Series by Holding Period
        ic_series_full = results['ic_series']
        for col in ic_series_full.columns:
            axes[0, 0].plot(ic_series_full.index, ic_series_full[col], label=col)
        axes[0, 0].set_title("Rank IC over Time")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("IC")
        axes[0, 0].grid(True, linestyle='--', alpha=0.3)
        axes[0, 0].legend(title="Period")
        
        # 2. Layer Returns (for the longest period)
        last_period = list(results['layer_returns'].keys())[-1]
        layer_ret = results['layer_returns'][last_period]
        layer_ret.plot(kind='bar', ax=axes[0, 1], color='coral')
        axes[0, 1].set_title(f"Mean Layer Returns ({last_period}d horizon)")
        axes[0, 1].set_ylabel("Return")
        
        # 3. IC Distribution (KDE for each holding period)
        ic_df = results['ic_series']
        all_values = ic_df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        if all_values.size > 0:
            x_min, x_max = all_values.min(), all_values.max()
            padding = 0.1 * (x_max - x_min) if x_max > x_min else 0.1
            x_grid = np.linspace(x_min - padding, x_max + padding, 200)

            for col in ic_df.columns:
                series = ic_df[col].dropna()
                if len(series) > 1:
                    kde = gaussian_kde(series.values)
                    axes[1, 0].plot(x_grid, kde(x_grid), label=col)

            axes[1, 0].set_title("IC Distribution (KDE by Period)")
            axes[1, 0].set_xlabel("IC")
            axes[1, 0].set_ylabel("Density")
            axes[1, 0].grid(True, linestyle='--', alpha=0.3)
            axes[1, 0].legend(title="Period")
        
        # 4. Factor Turnover (Rank Autocorrelation)
        turnover_series = results['turnover_series'].dropna()
        if not turnover_series.empty:
            turnover_series.plot(ax=axes[1, 1], color='green')
            axes[1, 1].set_title("Factor Turnover (Rank Autocorrelation)")
            axes[1, 1].axhline(turnover_series.mean(), color='red', linestyle='--', label=f'Mean: {turnover_series.mean():.4f}')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, "Insufficient data for turnover", ha='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close(fig)
