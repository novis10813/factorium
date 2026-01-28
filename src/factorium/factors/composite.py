"""
Multi-factor composition support.

Allows combining multiple factors using weighted combinations.
"""

from typing import List, Dict, Optional
import polars as pl

from .core import Factor


class CompositeFactor:
    """
    Weighted combination of multiple factors.
    
    Args:
        factors: List of Factor objects
        weights: Optional weights (defaults to equal weight)
        name: Name for the composite factor
    
    Example:
        >>> momentum = data["close"].ts_return(20)
        >>> value = data["volume"].cs_rank()
        >>> composite = CompositeFactor([momentum, value], weights=[0.6, 0.4])
        >>> signal = composite.to_factor()
    """
    
    def __init__(
        self,
        factors: List[Factor],
        weights: Optional[List[float]] = None,
        name: str = "composite",
    ):
        if len(factors) == 0:
            raise ValueError("At least one factor required")
        
        if weights is None:
            weights = [1.0 / len(factors)] * len(factors)
        
        if len(weights) != len(factors):
            raise ValueError("Number of weights must match number of factors")
        
        self.factors = factors
        self.weights = weights
        self.name = name
    
    def to_factor(self) -> Factor:
        """
        Combine factors into a single Factor.
        
        Returns:
            Factor representing weighted combination
        """
        # Start with first factor
        result = self.factors[0].lazy.with_columns(
            (pl.col("factor") * self.weights[0]).alias("weighted")
        )
        
        # Add remaining factors
        for i, factor in enumerate(self.factors[1:], start=1):
            factor_df = factor.lazy.with_columns(
                (pl.col("factor") * self.weights[i]).alias(f"weighted_{i}")
            )
            result = result.join(
                factor_df.select(["start_time", "end_time", "symbol", f"weighted_{i}"]),
                on=["start_time", "end_time", "symbol"],
                how="left",
            )
            result = result.with_columns(
                (pl.col("weighted") + pl.col(f"weighted_{i}").fill_null(0)).alias("weighted")
            )
        
        # Collect and create Factor
        result_df = result.select(["start_time", "end_time", "symbol", "weighted"]).collect()
        result_df = result_df.rename({"weighted": "factor"})
        
        return Factor(result_df, name=self.name)
