"""Alpha Pipeline: three-stage signal-to-weight transformation.

Stage 1: Normalize (raw alpha → known range)
Stage 2: Allocate (normalized signal → weights with invariants)
Stage 3: Constrain + renormalize (apply bounds, restore invariants)
"""

import polars as pl

from .allocators import MarketNeutralAllocator, WeightAllocator
from .constraints import WeightConstraint
from .normalizers import Normalizer, RawNormalizer


class AlphaPipeline:
    """Complete signal-to-weight transformation pipeline."""

    def __init__(
        self,
        normalizer: Normalizer | None = None,
        allocator: WeightAllocator | None = None,
        constraints: list[WeightConstraint] | None = None,
    ):
        self.normalizer = normalizer or RawNormalizer()
        self.allocator = allocator or MarketNeutralAllocator()
        self.constraints = constraints or []

    def transform(
        self,
        df: pl.DataFrame,
        signal_col: str,
        group_col: str = "end_time",
    ) -> pl.DataFrame:
        """Transform raw signal to constrained portfolio weights.

        Args:
            df: DataFrame containing the signal column
            signal_col: Name of the signal column
            group_col: Column to group by for cross-sectional operations

        Returns:
            DataFrame with 'weight' column added
        """
        # Stage 1: Normalize
        df = self.normalizer.normalize(df, signal_col, group_col)

        # Stage 2: Allocate
        df = self.allocator.allocate(df, signal_col, group_col)

        # Stage 3: Constrain + renormalize
        for constraint in self.constraints:
            df = constraint.apply(df)
        if self.constraints:
            df = self.allocator.renormalize(df, group_col)

        return df
