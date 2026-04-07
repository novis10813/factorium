# Alpha Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `neutralization` + `constraints` parameters in `VectorizedBacktester` with a three-stage `AlphaPipeline` (Normalize → Allocate → Constrain).

**Architecture:** New files `normalizers.py`, `allocators.py`, `pipeline.py` under `src/factorium/backtest/`. `VectorizedBacktester` delegates weight calculation to `AlphaPipeline.transform()`. `ResearchSession.backtest()` updated to accept `pipeline` parameter. Breaking change for 0.4.0.

**Tech Stack:** Python 3.10+, Polars (DataFrame operations), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/factorium/backtest/normalizers.py` | Create | `Normalizer` ABC + `RawNormalizer`, `RankNormalizer`, `ZScoreNormalizer`, `MinMaxNormalizer` |
| `src/factorium/backtest/allocators.py` | Create | `WeightAllocator` ABC + `MarketNeutralAllocator`, `LongOnlyAllocator`, `TopNAllocator` |
| `src/factorium/backtest/pipeline.py` | Create | `AlphaPipeline` orchestrating the three stages |
| `src/factorium/backtest/vectorized.py` | Modify | Replace `neutralization`/`constraints` params with `pipeline`, simplify `_calculate_weights()` |
| `src/factorium/backtest/utils.py` | Modify | Delete `neutralize_weights_polars()` and `renormalize_weights()` |
| `src/factorium/backtest/__init__.py` | Modify | Update exports |
| `src/factorium/research/session.py` | Modify | Replace `neutralization` param with `pipeline` in `backtest()` |
| `tests/backtest/test_normalizers.py` | Create | Tests for all Normalizer implementations |
| `tests/backtest/test_allocators.py` | Create | Tests for all Allocator implementations |
| `tests/backtest/test_pipeline.py` | Create | Tests for AlphaPipeline integration |
| `tests/backtest/test_vectorized.py` | Modify | Update to use `pipeline=` instead of `neutralization=`/`constraints=` |
| `tests/backtest/test_backtester.py` | Modify | Remove `TestRenormalizeWeights`, update integration tests |
| `tests/backtest/test_utils.py` | Modify | Remove `test_neutralize_weights_polars` |
| `docs/user-guide/backtest.md` | Modify | Update docs for pipeline API |

---

### Task 1: Create Normalizer ABC and implementations

**Files:**
- Create: `src/factorium/backtest/normalizers.py`
- Create: `tests/backtest/test_normalizers.py`

- [ ] **Step 1: Write failing tests for RawNormalizer**

```python
# tests/backtest/test_normalizers.py
import polars as pl
import pytest

from factorium.backtest.normalizers import (
    MinMaxNormalizer,
    Normalizer,
    RankNormalizer,
    RawNormalizer,
    ZScoreNormalizer,
)


class TestRawNormalizer:
    def test_passthrough(self):
        """RawNormalizer should not modify the signal."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [0.8, -0.5, 2.0],
        })
        result = RawNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"].to_list() == [0.8, -0.5, 2.0]

    def test_null_passthrough(self):
        """RawNormalizer should preserve nulls."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [0.8, None, 2.0],
        })
        result = RawNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"].to_list() == [0.8, None, 2.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_normalizers.py::TestRawNormalizer -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'factorium.backtest.normalizers'`

- [ ] **Step 3: Implement Normalizer ABC and RawNormalizer**

```python
# src/factorium/backtest/normalizers.py
"""Signal normalizers for the Alpha Pipeline.

Each normalizer transforms raw alpha signals to a known range
before weight allocation.
"""

from abc import ABC, abstractmethod

import polars as pl


class Normalizer(ABC):
    """Transform raw alpha to a known range."""

    @abstractmethod
    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        """Normalize signal_col in-place (overwrite the column).

        Args:
            df: DataFrame containing the signal column
            signal_col: Name of the signal column to normalize
            group_col: Column to group by for cross-sectional operations

        Returns:
            DataFrame with signal_col replaced by normalized values
        """
        ...


class RawNormalizer(Normalizer):
    """Pass-through normalizer. No transformation applied."""

    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        return df
```

- [ ] **Step 4: Run tests to verify RawNormalizer passes**

Run: `uv run pytest tests/backtest/test_normalizers.py::TestRawNormalizer -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for RankNormalizer**

Append to `tests/backtest/test_normalizers.py`:

```python
class TestRankNormalizer:
    def test_output_range_zero_to_one(self):
        """RankNormalizer output should be in [0, 1]."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [10.0, 30.0, 20.0, 40.0],
        })
        result = RankNormalizer().normalize(df, "signal", "end_time")
        values = result["signal"]
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_preserves_ranking_order(self):
        """Higher signal should get higher rank value."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, 3.0, 2.0],
        })
        result = RankNormalizer().normalize(df, "signal", "end_time")
        vals = result["signal"].to_list()
        # B (3.0) > C (2.0) > A (1.0), so rank(B) > rank(C) > rank(A)
        assert vals[1] > vals[2] > vals[0]

    def test_cross_sectional_per_group(self):
        """Ranking should be independent per group."""
        df = pl.DataFrame({
            "end_time": [1000, 1000, 2000, 2000],
            "symbol": ["A", "B", "A", "B"],
            "signal": [10.0, 20.0, 50.0, 5.0],
        })
        result = RankNormalizer().normalize(df, "signal", "end_time")
        # Group 1000: B > A, Group 2000: A > B
        g1 = result.filter(pl.col("end_time") == 1000)["signal"].to_list()
        g2 = result.filter(pl.col("end_time") == 2000)["signal"].to_list()
        assert g1[1] > g1[0]  # B > A in group 1000
        assert g2[0] > g2[1]  # A > B in group 2000

    def test_null_signal_stays_null(self):
        """Null signals should remain null after ranking."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, None, 3.0],
        })
        result = RankNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"][1] is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_normalizers.py::TestRankNormalizer -v`
Expected: FAIL with `ImportError` (RankNormalizer not yet implemented)

- [ ] **Step 7: Implement RankNormalizer**

Append to `src/factorium/backtest/normalizers.py`:

```python
class RankNormalizer(Normalizer):
    """Cross-sectional rank normalization to [0, 1]."""

    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        rank = pl.col(signal_col).rank().over(group_col)
        count = pl.col(signal_col).count().over(group_col)
        return df.with_columns((rank / count).alias(signal_col))
```

- [ ] **Step 8: Run tests to verify RankNormalizer passes**

Run: `uv run pytest tests/backtest/test_normalizers.py::TestRankNormalizer -v`
Expected: PASS

- [ ] **Step 9: Write failing tests for ZScoreNormalizer**

Append to `tests/backtest/test_normalizers.py`:

```python
class TestZScoreNormalizer:
    def test_mean_near_zero(self):
        """Cross-sectional z-score mean should be ~0."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = ZScoreNormalizer().normalize(df, "signal", "end_time")
        assert abs(result["signal"].mean()) < 1e-10

    def test_std_near_one(self):
        """Cross-sectional z-score std should be ~1."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = ZScoreNormalizer().normalize(df, "signal", "end_time")
        assert abs(result["signal"].std() - 1.0) < 0.1

    def test_zero_std_produces_null(self):
        """When all signals are identical (std=0), output should be null."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [5.0, 5.0, 5.0],
        })
        result = ZScoreNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"].null_count() == 3

    def test_null_signal_stays_null(self):
        """Null signals should remain null after z-score."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, None, 3.0],
        })
        result = ZScoreNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"][1] is None
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_normalizers.py::TestZScoreNormalizer -v`
Expected: FAIL

- [ ] **Step 11: Implement ZScoreNormalizer**

Append to `src/factorium/backtest/normalizers.py`:

```python
class ZScoreNormalizer(Normalizer):
    """Cross-sectional z-score normalization to approximately [-3, 3]."""

    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        mean = pl.col(signal_col).mean().over(group_col)
        std = pl.col(signal_col).std().over(group_col)
        return df.with_columns(
            ((pl.col(signal_col) - mean) / std)
            .fill_nan(None)
            .alias(signal_col)
        )
```

- [ ] **Step 12: Run tests to verify ZScoreNormalizer passes**

Run: `uv run pytest tests/backtest/test_normalizers.py::TestZScoreNormalizer -v`
Expected: PASS

- [ ] **Step 13: Write failing tests for MinMaxNormalizer**

Append to `tests/backtest/test_normalizers.py`:

```python
class TestMinMaxNormalizer:
    def test_output_range_zero_to_one(self):
        """MinMaxNormalizer output should be in [0, 1]."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [10.0, 30.0, 20.0, 40.0],
        })
        result = MinMaxNormalizer().normalize(df, "signal", "end_time")
        values = result["signal"]
        assert values.min() >= -1e-10
        assert values.max() <= 1.0 + 1e-10

    def test_min_maps_to_zero_max_maps_to_one(self):
        """Min value should map to 0, max to 1."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [10.0, 20.0, 30.0],
        })
        result = MinMaxNormalizer().normalize(df, "signal", "end_time")
        vals = result["signal"].to_list()
        assert abs(vals[0] - 0.0) < 1e-10  # min -> 0
        assert abs(vals[2] - 1.0) < 1e-10  # max -> 1

    def test_zero_range_produces_null(self):
        """When all signals are identical (range=0), output should be null."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [5.0, 5.0, 5.0],
        })
        result = MinMaxNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"].null_count() == 3

    def test_cross_sectional_per_group(self):
        """MinMax should be independent per group."""
        df = pl.DataFrame({
            "end_time": [1000, 1000, 2000, 2000],
            "symbol": ["A", "B", "A", "B"],
            "signal": [10.0, 20.0, 100.0, 200.0],
        })
        result = MinMaxNormalizer().normalize(df, "signal", "end_time")
        g1 = result.filter(pl.col("end_time") == 1000)["signal"].to_list()
        g2 = result.filter(pl.col("end_time") == 2000)["signal"].to_list()
        # Both groups should have min=0, max=1
        assert abs(g1[0] - 0.0) < 1e-10
        assert abs(g1[1] - 1.0) < 1e-10
        assert abs(g2[0] - 0.0) < 1e-10
        assert abs(g2[1] - 1.0) < 1e-10
```

- [ ] **Step 14: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_normalizers.py::TestMinMaxNormalizer -v`
Expected: FAIL

- [ ] **Step 15: Implement MinMaxNormalizer**

Append to `src/factorium/backtest/normalizers.py`:

```python
from ..constants import EPSILON


class MinMaxNormalizer(Normalizer):
    """Cross-sectional min-max normalization to [0, 1]."""

    def normalize(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        min_val = pl.col(signal_col).min().over(group_col)
        max_val = pl.col(signal_col).max().over(group_col)
        denom = max_val - min_val
        return df.with_columns(
            pl.when(denom.abs() <= EPSILON)
            .then(pl.lit(None))
            .otherwise((pl.col(signal_col) - min_val) / denom)
            .alias(signal_col)
        )
```

Note: The `EPSILON` import should be at the top of the file. Move `from ..constants import EPSILON` to the imports section.

- [ ] **Step 16: Run all normalizer tests**

Run: `uv run pytest tests/backtest/test_normalizers.py -v`
Expected: All PASS

- [ ] **Step 17: Commit**

```bash
git add src/factorium/backtest/normalizers.py tests/backtest/test_normalizers.py
git commit -m "feat(backtest): add Normalizer ABC and four implementations

RawNormalizer (pass-through), RankNormalizer ([0,1] rank),
ZScoreNormalizer (z-score), MinMaxNormalizer ([0,1] min-max)."
```

---

### Task 2: Create WeightAllocator ABC and MarketNeutralAllocator

**Files:**
- Create: `src/factorium/backtest/allocators.py`
- Create: `tests/backtest/test_allocators.py`

- [ ] **Step 1: Write failing tests for MarketNeutralAllocator**

```python
# tests/backtest/test_allocators.py
import polars as pl
import pytest

from factorium.backtest.allocators import (
    LongOnlyAllocator,
    MarketNeutralAllocator,
    TopNAllocator,
    WeightAllocator,
)


class TestMarketNeutralAllocator:
    def test_weights_sum_to_zero(self):
        """Market neutral weights should sum to zero per group."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [0.8, 0.5, 0.2, 0.1],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        assert abs(result["weight"].sum()) < 1e-10

    def test_abs_weights_sum_to_one(self):
        """Absolute weights should sum to one per group."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [0.8, 0.5, 0.2, 0.1],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10

    def test_null_signal_gets_zero_weight(self):
        """Null signals should produce zero weight."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, None, 3.0],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        assert result["weight"][1] == 0.0

    def test_identical_signals_produce_zero_weights(self):
        """Identical signals should all be zero after demeaning."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [5.0, 5.0, 5.0],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        assert result["weight"].abs().sum() < 1e-10

    def test_multiple_groups(self):
        """Each group should be independently neutralized."""
        df = pl.DataFrame({
            "end_time": [1000, 1000, 2000, 2000],
            "symbol": ["A", "B", "A", "B"],
            "signal": [10.0, 20.0, 5.0, 15.0],
        })
        result = MarketNeutralAllocator().allocate(df, "signal", "end_time")
        for t in [1000, 2000]:
            subset = result.filter(pl.col("end_time") == t)["weight"]
            assert abs(subset.sum()) < 1e-10
            assert abs(subset.abs().sum() - 1.0) < 1e-10

    def test_renormalize_restores_invariants(self):
        """renormalize should restore sum=0, abs_sum=1 after perturbation."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.3, 0.1, -0.05, -0.1],  # broken invariants
        })
        result = MarketNeutralAllocator().renormalize(df, "end_time")
        assert abs(result["weight"].sum()) < 1e-10
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10

    def test_renormalize_all_zero_stays_zero(self):
        """All-zero weights should stay zero after renormalize."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.0, 0.0, 0.0],
        })
        result = MarketNeutralAllocator().renormalize(df, "end_time")
        assert result["weight"].abs().sum() < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_allocators.py::TestMarketNeutralAllocator -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement WeightAllocator ABC and MarketNeutralAllocator**

```python
# src/factorium/backtest/allocators.py
"""Weight allocators for the Alpha Pipeline.

Each allocator converts normalized signals to portfolio weights
satisfying specific invariants (e.g., market neutral, long-only).
"""

from abc import ABC, abstractmethod

import polars as pl

from ..constants import EPSILON


class WeightAllocator(ABC):
    """Convert normalized signal to portfolio weights."""

    @abstractmethod
    def allocate(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        """Add a 'weight' column satisfying the allocator's invariants.

        Args:
            df: DataFrame containing the signal column
            signal_col: Name of the normalized signal column
            group_col: Column to group by for cross-sectional operations

        Returns:
            DataFrame with 'weight' column added
        """
        ...

    @abstractmethod
    def renormalize(self, df: pl.DataFrame, group_col: str) -> pl.DataFrame:
        """Restore weight invariants after constraint application.

        Args:
            df: DataFrame with 'weight' column
            group_col: Column to group by

        Returns:
            DataFrame with renormalized weights
        """
        ...


class MarketNeutralAllocator(WeightAllocator):
    """Dollar-neutral allocator: sum(w)=0, sum(|w|)=1."""

    def allocate(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        demeaned = pl.col(signal_col) - pl.col(signal_col).mean().over(group_col)
        abs_sum = demeaned.abs().sum().over(group_col)
        weight = (demeaned / abs_sum).fill_nan(0.0).fill_null(0.0)
        return df.with_columns(weight.alias("weight"))

    def renormalize(self, df: pl.DataFrame, group_col: str) -> pl.DataFrame:
        df = df.with_columns(
            (pl.col("weight") - pl.col("weight").mean().over(group_col)).alias("weight")
        )
        abs_sum = pl.col("weight").abs().sum().over(group_col)
        return df.with_columns(
            pl.when(abs_sum > EPSILON)
            .then(pl.col("weight") / abs_sum)
            .otherwise(0.0)
            .alias("weight")
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_allocators.py::TestMarketNeutralAllocator -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/factorium/backtest/allocators.py tests/backtest/test_allocators.py
git commit -m "feat(backtest): add WeightAllocator ABC and MarketNeutralAllocator

Dollar-neutral: sum(w)=0, sum(|w|)=1 per cross-section."
```

---

### Task 3: Add LongOnlyAllocator

**Files:**
- Modify: `src/factorium/backtest/allocators.py`
- Modify: `tests/backtest/test_allocators.py`

- [ ] **Step 1: Write failing tests for LongOnlyAllocator**

Append to `tests/backtest/test_allocators.py`:

```python
class TestLongOnlyAllocator:
    def test_weights_sum_to_one(self):
        """Long-only weights should sum to one."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, 2.0, 3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert abs(result["weight"].sum() - 1.0) < 1e-10

    def test_all_weights_non_negative(self):
        """All weights should be >= 0."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [1.0, 2.0, 3.0, -1.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert (result["weight"] >= -1e-10).all()

    def test_negative_signals_get_zero_weight(self):
        """Negative signals should get zero weight."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, -2.0, 3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert result["weight"][1] == 0.0

    def test_proportional_to_signal(self):
        """Weights should be proportional to positive signal values."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, 2.0, 3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        weights = result["weight"].to_list()
        # B should be 2x A, C should be 3x A
        assert abs(weights[1] / weights[0] - 2.0) < 1e-10
        assert abs(weights[2] / weights[0] - 3.0) < 1e-10

    def test_all_negative_signals_produce_zero_weights(self):
        """If all signals are negative, all weights should be zero."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [-1.0, -2.0, -3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert result["weight"].abs().sum() < 1e-10

    def test_null_signal_gets_zero_weight(self):
        """Null signals should produce zero weight."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, None, 3.0],
        })
        result = LongOnlyAllocator().allocate(df, "signal", "end_time")
        assert result["weight"][1] == 0.0

    def test_renormalize_clips_negatives_and_sums_to_one(self):
        """renormalize should clip negatives and scale to sum=1."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.5, 0.3, -0.1],
        })
        result = LongOnlyAllocator().renormalize(df, "end_time")
        assert (result["weight"] >= -1e-10).all()
        assert abs(result["weight"].sum() - 1.0) < 1e-10

    def test_renormalize_all_zero_stays_zero(self):
        """All-zero weights should stay zero after renormalize."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.0, 0.0, 0.0],
        })
        result = LongOnlyAllocator().renormalize(df, "end_time")
        assert result["weight"].abs().sum() < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_allocators.py::TestLongOnlyAllocator -v`
Expected: FAIL (LongOnlyAllocator not yet implemented)

- [ ] **Step 3: Implement LongOnlyAllocator**

Append to `src/factorium/backtest/allocators.py`:

```python
class LongOnlyAllocator(WeightAllocator):
    """Long-only allocator: sum(w)=1, all w>=0. Only positive signals get weight."""

    def allocate(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        positive = (
            pl.when(pl.col(signal_col) > 0)
            .then(pl.col(signal_col))
            .otherwise(0.0)
        )
        w_sum = positive.sum().over(group_col)
        weight = (
            pl.when(w_sum > EPSILON)
            .then(positive / w_sum)
            .otherwise(0.0)
        )
        return df.with_columns(weight.fill_null(0.0).alias("weight"))

    def renormalize(self, df: pl.DataFrame, group_col: str) -> pl.DataFrame:
        df = df.with_columns(
            pl.when(pl.col("weight") < 0.0)
            .then(0.0)
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        w_sum = pl.col("weight").sum().over(group_col)
        return df.with_columns(
            pl.when(w_sum > EPSILON)
            .then(pl.col("weight") / w_sum)
            .otherwise(0.0)
            .alias("weight")
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_allocators.py::TestLongOnlyAllocator -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/factorium/backtest/allocators.py tests/backtest/test_allocators.py
git commit -m "feat(backtest): add LongOnlyAllocator

Long-only: sum(w)=1, w>=0. Only positive signals receive weight."
```

---

### Task 4: Add TopNAllocator

**Files:**
- Modify: `src/factorium/backtest/allocators.py`
- Modify: `tests/backtest/test_allocators.py`

- [ ] **Step 1: Write failing tests for TopNAllocator**

Append to `tests/backtest/test_allocators.py`:

```python
class TestTopNAllocator:
    def test_long_only_top_n_equal_weight(self):
        """Top N long-only should give equal weight 1/N to top N."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [1.0, 5.0, 3.0, 4.0, 2.0],
        })
        result = TopNAllocator(n=2).allocate(df, "signal", "end_time")
        weights = result.sort("symbol")["weight"].to_list()
        # B (5.0) and D (4.0) are top 2
        assert weights[1] == pytest.approx(0.5)  # B
        assert weights[3] == pytest.approx(0.5)  # D
        assert weights[0] == 0.0  # A
        assert weights[2] == 0.0  # C
        assert weights[4] == 0.0  # E

    def test_long_short_top_n(self):
        """Long-short mode: top N get +1/N, bottom N get -1/N."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [1.0, 5.0, 3.0, 4.0, 2.0],
        })
        result = TopNAllocator(n=2, long_short=True).allocate(df, "signal", "end_time")
        weights = result.sort("symbol")["weight"].to_list()
        # Top 2: B (5.0), D (4.0) → +0.5
        # Bottom 2: A (1.0), E (2.0) → -0.5
        assert weights[1] == pytest.approx(0.5)   # B
        assert weights[3] == pytest.approx(0.5)   # D
        assert weights[0] == pytest.approx(-0.5)  # A
        assert weights[4] == pytest.approx(-0.5)  # E
        assert weights[2] == 0.0                   # C (middle)

    def test_long_short_weights_sum_to_zero(self):
        """Long-short weights should sum to zero."""
        df = pl.DataFrame({
            "end_time": [1000] * 6,
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "signal": [1.0, 6.0, 3.0, 5.0, 2.0, 4.0],
        })
        result = TopNAllocator(n=2, long_short=True).allocate(df, "signal", "end_time")
        assert abs(result["weight"].sum()) < 1e-10

    def test_long_only_weights_sum_to_one(self):
        """Long-only top-N weights should sum to one."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [1.0, 5.0, 3.0, 4.0, 2.0],
        })
        result = TopNAllocator(n=3).allocate(df, "signal", "end_time")
        non_zero = result.filter(pl.col("weight") != 0.0)
        assert abs(non_zero["weight"].sum() - 1.0) < 1e-10

    def test_null_signal_excluded(self):
        """Null signals should not be selected in top N."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [1.0, None, 3.0, 2.0],
        })
        result = TopNAllocator(n=2).allocate(df, "signal", "end_time")
        assert result.filter(pl.col("symbol") == "B")["weight"][0] == 0.0

    def test_renormalize_restores_equal_weight(self):
        """renormalize should restore equal-weight among non-zero positions."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.4, 0.3, 0.0, 0.0],  # perturbed from equal
        })
        result = TopNAllocator(n=2).renormalize(df, "end_time")
        non_zero = result.filter(pl.col("weight") != 0.0)["weight"].to_list()
        assert len(non_zero) == 2
        assert abs(non_zero[0] - non_zero[1]) < 1e-10
        assert abs(sum(non_zero) - 1.0) < 1e-10

    def test_renormalize_long_short_equal_weight(self):
        """renormalize long-short should restore equal-weight on both sides."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.4, 0.3, -0.2, -0.1],  # perturbed
        })
        result = TopNAllocator(n=2, long_short=True).renormalize(df, "end_time")
        pos = result.filter(pl.col("weight") > 0)["weight"].to_list()
        neg = result.filter(pl.col("weight") < 0)["weight"].to_list()
        assert abs(pos[0] - pos[1]) < 1e-10
        assert abs(neg[0] - neg[1]) < 1e-10
        assert abs(sum(pos) + sum(neg)) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_allocators.py::TestTopNAllocator -v`
Expected: FAIL

- [ ] **Step 3: Implement TopNAllocator**

Append to `src/factorium/backtest/allocators.py`:

```python
class TopNAllocator(WeightAllocator):
    """Equal-weight top N allocator. Optionally long-short (top N long, bottom N short)."""

    def __init__(self, n: int, long_short: bool = False):
        self.n = n
        self.long_short = long_short

    def allocate(
        self, df: pl.DataFrame, signal_col: str, group_col: str
    ) -> pl.DataFrame:
        rank = pl.col(signal_col).rank(descending=True).over(group_col)
        count = pl.col(signal_col).count().over(group_col)

        long_w = pl.lit(1.0 / self.n)

        if self.long_short:
            short_w = pl.lit(-1.0 / self.n)
            weight = (
                pl.when(rank <= self.n)
                .then(long_w)
                .when(rank > count - self.n)
                .then(short_w)
                .otherwise(0.0)
            )
        else:
            weight = (
                pl.when(rank <= self.n)
                .then(long_w)
                .otherwise(0.0)
            )

        return df.with_columns(weight.fill_null(0.0).alias("weight"))

    def renormalize(self, df: pl.DataFrame, group_col: str) -> pl.DataFrame:
        if self.long_short:
            # Count non-zero positions on each side per group
            pos_count = (
                pl.col("weight")
                .filter(pl.col("weight") > EPSILON)
                .count()
                .over(group_col)
            )
            neg_count = (
                pl.col("weight")
                .filter(pl.col("weight") < -EPSILON)
                .count()
                .over(group_col)
            )
            weight = (
                pl.when(pl.col("weight") > EPSILON)
                .then(1.0 / pos_count)
                .when(pl.col("weight") < -EPSILON)
                .then(-1.0 / neg_count)
                .otherwise(0.0)
            )
        else:
            pos_count = (
                pl.col("weight")
                .filter(pl.col("weight") > EPSILON)
                .count()
                .over(group_col)
            )
            weight = (
                pl.when(pl.col("weight") > EPSILON)
                .then(1.0 / pos_count)
                .otherwise(0.0)
            )

        return df.with_columns(weight.alias("weight"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_allocators.py::TestTopNAllocator -v`
Expected: PASS

- [ ] **Step 5: Run all allocator tests**

Run: `uv run pytest tests/backtest/test_allocators.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/factorium/backtest/allocators.py tests/backtest/test_allocators.py
git commit -m "feat(backtest): add TopNAllocator

Equal-weight top N, with optional long-short mode."
```

---

### Task 5: Create AlphaPipeline

**Files:**
- Create: `src/factorium/backtest/pipeline.py`
- Create: `tests/backtest/test_pipeline.py`

- [ ] **Step 1: Write failing tests for AlphaPipeline**

```python
# tests/backtest/test_pipeline.py
import polars as pl
import pytest

from factorium.backtest.allocators import LongOnlyAllocator, MarketNeutralAllocator
from factorium.backtest.constraints import MaxPositionConstraint
from factorium.backtest.normalizers import RankNormalizer, RawNormalizer
from factorium.backtest.pipeline import AlphaPipeline


class TestAlphaPipelineDefaults:
    def test_default_pipeline_is_raw_market_neutral(self):
        """Default pipeline should be RawNormalizer + MarketNeutralAllocator."""
        pipe = AlphaPipeline()
        assert isinstance(pipe.normalizer, RawNormalizer)
        assert isinstance(pipe.allocator, MarketNeutralAllocator)
        assert pipe.constraints == []

    def test_default_produces_market_neutral_weights(self):
        """Default pipeline output should satisfy market-neutral invariants."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [0.8, 0.5, 0.2, 0.1],
        })
        result = AlphaPipeline().transform(df, "signal")
        assert abs(result["weight"].sum()) < 1e-10
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10


class TestAlphaPipelineWithNormalizer:
    def test_rank_then_long_only(self):
        """RankNormalizer + LongOnlyAllocator should produce valid weights."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [-10.0, 50.0, 20.0],
        })
        pipe = AlphaPipeline(
            normalizer=RankNormalizer(),
            allocator=LongOnlyAllocator(),
        )
        result = pipe.transform(df, "signal")
        # After rank normalization, all values are in [0,1] (positive)
        # so LongOnly should give all assets weight
        assert abs(result["weight"].sum() - 1.0) < 1e-10
        assert (result["weight"] >= -1e-10).all()


class TestAlphaPipelineWithConstraints:
    def test_constraint_then_renormalize(self):
        """Constraints should be applied, then renormalize restores invariants."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [10.0, 1.0, 1.0, 1.0],
        })
        pipe = AlphaPipeline(
            normalizer=RawNormalizer(),
            allocator=MarketNeutralAllocator(),
            constraints=[MaxPositionConstraint(max_weight=0.3)],
        )
        result = pipe.transform(df, "signal")
        # After constraint + renormalize, invariants should hold
        assert abs(result["weight"].sum()) < 1e-10
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10
        # Max weight should be capped (within renormalize tolerance)
        assert result["weight"].abs().max() <= 0.5 + 1e-10

    def test_no_constraints_skips_renormalize(self):
        """Without constraints, renormalize should not be called."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, 2.0, 3.0],
        })
        pipe = AlphaPipeline(
            normalizer=RawNormalizer(),
            allocator=MarketNeutralAllocator(),
        )
        result = pipe.transform(df, "signal")
        # Should still satisfy invariants from allocate alone
        assert abs(result["weight"].sum()) < 1e-10
        assert abs(result["weight"].abs().sum() - 1.0) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backtest/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'factorium.backtest.pipeline'`

- [ ] **Step 3: Implement AlphaPipeline**

```python
# src/factorium/backtest/pipeline.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_pipeline.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/factorium/backtest/pipeline.py tests/backtest/test_pipeline.py
git commit -m "feat(backtest): add AlphaPipeline

Three-stage pipeline: Normalize → Allocate → Constrain + Renormalize."
```

---

### Task 6: Integrate pipeline into VectorizedBacktester

**Files:**
- Modify: `src/factorium/backtest/vectorized.py`
- Modify: `tests/backtest/test_vectorized.py`

- [ ] **Step 1: Update test_vectorized.py to use pipeline API**

Replace the contents of `tests/backtest/test_vectorized.py` with:

```python
"""Tests for VectorizedBacktester."""

import pytest
import polars as pl
import numpy as np

from factorium import AggBar
from factorium.backtest.allocators import LongOnlyAllocator, MarketNeutralAllocator
from factorium.backtest.constraints import MaxPositionConstraint, LongOnlyConstraint
from factorium.backtest.normalizers import RankNormalizer, RawNormalizer
from factorium.backtest.pipeline import AlphaPipeline
from factorium.backtest.vectorized import VectorizedBacktester, BacktestResult


class TestVectorizedBacktesterInit:
    """Tests for VectorizedBacktester initialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 50, 3600000))

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": price * 0.99,
                        "high": price * 1.01,
                        "low": price * 0.98,
                        "close": price,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_init_default_pipeline(self, sample_data):
        """Default pipeline should be RawNormalizer + MarketNeutralAllocator."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        assert isinstance(bt.pipeline.normalizer, RawNormalizer)
        assert isinstance(bt.pipeline.allocator, MarketNeutralAllocator)

    def test_init_with_custom_pipeline(self, sample_data):
        """Should accept custom pipeline."""
        signal = sample_data["close"].cs_rank()
        pipe = AlphaPipeline(
            normalizer=RankNormalizer(),
            allocator=LongOnlyAllocator(),
        )
        bt = VectorizedBacktester(prices=sample_data, signal=signal, pipeline=pipe)
        assert isinstance(bt.pipeline.normalizer, RankNormalizer)

    def test_run_returns_result(self, sample_data):
        """run() should return BacktestResult."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert result.returns is not None
        assert result.metrics is not None

    def test_equity_curve_is_polars_dataframe(self, sample_data):
        """equity_curve should be Polars DataFrame."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert isinstance(result.equity_curve, pl.DataFrame)
        assert "end_time" in result.equity_curve.columns
        assert "total_value" in result.equity_curve.columns

    def test_total_value_positive(self, sample_data):
        """Total value should always be positive."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert result.equity_curve["total_value"].min() > 0


class TestWeightCalculation:
    """Tests for weight calculation in VectorizedBacktester."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known signals."""
        timestamps = [1704067200000, 1704070800000, 1704074400000]

        rows = []
        for ts in timestamps:
            for symbol, signal in [("A", 0.8), ("B", 0.5), ("C", 0.2), ("D", -0.1)]:
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": 100.0,
                        "high": 100.0,
                        "low": 100.0,
                        "close": 100.0,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_market_neutral_weights_sum_to_zero(self, sample_data):
        """Market neutral weights should sum to zero."""
        signal = sample_data["close"]

        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
            pipeline=AlphaPipeline(
                allocator=MarketNeutralAllocator(),
            ),
        )

        combined = bt._prepare_data()
        weighted = bt._calculate_weights(combined)

        weight_sums = weighted.group_by("end_time").agg(pl.col("weight").sum().alias("weight_sum"))
        assert weight_sums["weight_sum"].abs().max() < 1e-10

    def test_long_only_weights_sum_to_one(self):
        """Long-only weights should sum to 1."""
        timestamps = [1704067200000, 1704070800000]

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol, price in [("A", 100.0), ("B", 50.0), ("C", 25.0)]:
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "close": price * (1 + 0.01 * i),
                        "open": price,
                        "high": price,
                        "low": price,
                        "volume": 1000.0,
                    }
                )

        data = AggBar(pl.DataFrame(rows))
        signal = data["close"].cs_rank()

        bt = VectorizedBacktester(
            prices=data,
            signal=signal,
            pipeline=AlphaPipeline(allocator=LongOnlyAllocator()),
        )

        combined = bt._prepare_data()
        weighted = bt._calculate_weights(combined)

        weight_sums = (
            weighted.filter(pl.col("weight") != 0).group_by("end_time").agg(pl.col("weight").sum().alias("weight_sum"))
        )

        for ws in weight_sums["weight_sum"].to_list():
            if ws > 0:
                assert abs(ws - 1.0) < 1e-10

    def test_calculate_weights_masked_assets_remain_zero_after_neutralize(self):
        timestamps = [1704067200000, 1704070800000, 1704074400000]
        rows = []
        for i, ts in enumerate(timestamps):
            for symbol, base_price, in_universe in [
                ("A", 100.0, True),
                ("B", 80.0, True),
                ("C", 60.0, False),
            ]:
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 1000.0,
                        "in_universe": in_universe,
                    }
                )

        prices = AggBar(pl.DataFrame(rows))
        signal = prices["close"].cs_rank()
        bt = VectorizedBacktester(
            prices=prices,
            signal=signal,
            pipeline=AlphaPipeline(allocator=MarketNeutralAllocator()),
            mask="in_universe",
        )

        combined = bt._prepare_data()
        weighted = bt._calculate_weights(combined)

        masked = weighted.filter(~pl.col("in_universe").fill_null(False))
        assert masked["weight"].abs().max() < 1e-10
        assert "_masked_signal" not in weighted.columns


class TestConstraintIntegration:
    """Tests for constraint integration via pipeline."""

    @pytest.fixture
    def sample_data(self):
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 20, 3600000))

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "close": price,
                        "open": price,
                        "high": price,
                        "low": price,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_max_position_constraint_via_pipeline(self, sample_data):
        """Should apply MaxPositionConstraint through pipeline."""
        signal = sample_data["close"].cs_rank()

        pipe = AlphaPipeline(
            allocator=MarketNeutralAllocator(),
            constraints=[MaxPositionConstraint(max_weight=0.1)],
        )

        bt = VectorizedBacktester(prices=sample_data, signal=signal, pipeline=pipe)
        result = bt.run()
        assert result.metrics is not None

    def test_long_only_constraint_via_pipeline(self, sample_data):
        """Should apply LongOnlyConstraint through pipeline."""
        signal = sample_data["close"].cs_rank()

        pipe = AlphaPipeline(
            allocator=MarketNeutralAllocator(),
            constraints=[LongOnlyConstraint()],
        )

        bt = VectorizedBacktester(prices=sample_data, signal=signal, pipeline=pipe)
        result = bt.run()
        assert result.metrics is not None
```

- [ ] **Step 2: Run updated tests to verify they fail**

Run: `uv run pytest tests/backtest/test_vectorized.py -v`
Expected: FAIL (VectorizedBacktester still uses old API)

- [ ] **Step 3: Modify VectorizedBacktester to accept pipeline**

Edit `src/factorium/backtest/vectorized.py`:

Replace the `__init__` signature and body — remove `neutralization` and `constraints` parameters, add `pipeline`:

```python
# In imports section, add:
from .pipeline import AlphaPipeline

# Replace __init__:
class VectorizedBacktester:
    """Vectorized backtester using Polars for high performance."""

    def __init__(
        self,
        prices: AggBar | pl.DataFrame,
        signal: Factor | pl.DataFrame,
        entry_price: str = "close",
        transaction_cost: float | tuple[float, float] = 0.0003,
        initial_capital: float = 10000.0,
        pipeline: AlphaPipeline | None = None,
        frequency: str = "1h",
        mask: str | None = None,
    ):
        """
        Initialize the vectorized backtester.

        Args:
            prices: AggBar or Polars DataFrame with OHLCV data
            signal: Factor or Polars DataFrame with signals
            entry_price: Column name in prices for execution price
            transaction_cost: Transaction cost as % of notional, or (buy, sell) tuple
            initial_capital: Starting portfolio value
            pipeline: AlphaPipeline for signal-to-weight conversion (default: Raw + MarketNeutral)
            frequency: Frequency string (e.g., "1h", "1d")
            mask: Column name in prices to filter tradeable universe
        """
        self.initial_capital = initial_capital

        if isinstance(transaction_cost, (int, float)):
            self.transaction_cost = (float(transaction_cost), float(transaction_cost))
        else:
            self.transaction_cost = transaction_cost

        self.entry_price = entry_price
        self.pipeline = pipeline or AlphaPipeline()
        self.frequency = frequency
        self.periods_per_year = frequency_to_periods_per_year(frequency)
        self._periods_per_year = self.periods_per_year
        self._mask = mask

        # Convert inputs to Polars DataFrames
        if isinstance(prices, AggBar):
            if entry_price not in prices.cols:
                raise ValueError(f"entry_price '{entry_price}' not found in prices")
            if mask is not None and mask not in prices.cols:
                raise ValueError(f"mask '{mask}' not found in prices")
            self.prices_df = prices.to_polars()
        else:
            self.prices_df = prices
            if entry_price not in prices.columns:
                raise ValueError(f"entry_price '{entry_price}' not found in prices")
            if mask is not None and mask not in prices.columns:
                raise ValueError(f"mask '{mask}' not found in prices")

        if isinstance(signal, Factor):
            self.signal_df = signal.lazy.collect()
        else:
            self.signal_df = signal

        self._result: BacktestResult | None = None
```

Replace `_calculate_weights` method:

```python
    def _calculate_weights(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate portfolio weights via pipeline."""
        signal_col = "prev_signal"
        if self._mask is not None:
            signal_col = "_masked_signal"
            df = df.with_columns(
                pl.when(pl.col(self._mask).fill_null(False))
                .then(pl.col("prev_signal"))
                .otherwise(None)
                .alias(signal_col)
            )

        df = self.pipeline.transform(df, signal_col, "end_time")

        if self._mask is not None:
            df = df.drop("_masked_signal")

        return df
```

- [ ] **Step 4: Run test_vectorized.py**

Run: `uv run pytest tests/backtest/test_vectorized.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/factorium/backtest/vectorized.py tests/backtest/test_vectorized.py
git commit -m "refactor(backtest): replace neutralization/constraints with pipeline

VectorizedBacktester now accepts AlphaPipeline parameter.
_calculate_weights() delegates to pipeline.transform()."
```

---

### Task 7: Update test_backtester.py and remaining test files

**Files:**
- Modify: `tests/backtest/test_backtester.py`
- Modify: `tests/backtest/test_utils.py`

- [ ] **Step 1: Update test_backtester.py**

In `tests/backtest/test_backtester.py`:

1. Remove `from factorium.backtest.utils import renormalize_weights` (line 20)
2. Delete the entire `TestRenormalizeWeights` class (lines 64-114)
3. Update `TestBacktester` and `TestVectorizedBacktesterIntegration` tests that use `neutralization=` to use `pipeline=` instead.

Updated imports:

```python
from factorium.backtest.allocators import LongOnlyAllocator, MarketNeutralAllocator
from factorium.backtest.pipeline import AlphaPipeline
```

Key test changes:

- `test_basic_backtest`: Replace `neutralization="market"` with `pipeline=AlphaPipeline(allocator=MarketNeutralAllocator())`
- `test_vectorized_produces_reasonable_equity`: Same replacement
- `test_single_symbol_backtest`: Replace `neutralization="none"` with `pipeline=AlphaPipeline(allocator=LongOnlyAllocator())`
- `test_missing_price_symbol_excluded`: Replace `neutralization="market"` with `pipeline=AlphaPipeline(allocator=MarketNeutralAllocator())`
- Remove `test_max_position_constraint` and `test_long_only_constraint` from `TestConstraintIntegration` in this file (already covered in test_vectorized.py)
- Tests that use `Backtester(prices=..., signal=..., neutralization=...)` need `pipeline=` instead
- Tests that use just `Backtester(prices=..., signal=...)` without neutralization don't need changes (default pipeline matches old default)

- [ ] **Step 2: Update test_utils.py**

In `tests/backtest/test_utils.py`:

1. Remove `neutralize_weights_polars` from the import (line 5)
2. Delete `test_neutralize_weights_polars` function (lines 9-24)

Updated import:

```python
from factorium.backtest.utils import safe_divide
```

- [ ] **Step 3: Run updated tests**

Run: `uv run pytest tests/backtest/test_backtester.py tests/backtest/test_utils.py -v`
Expected: FAIL (utils still has old functions, tests reference removed items)

- [ ] **Step 4: Commit test updates**

```bash
git add tests/backtest/test_backtester.py tests/backtest/test_utils.py
git commit -m "test(backtest): update tests for pipeline API

Remove tests for deleted utils functions. Update integration tests
to use pipeline= parameter."
```

---

### Task 8: Clean up utils.py and __init__.py

**Files:**
- Modify: `src/factorium/backtest/utils.py`
- Modify: `src/factorium/backtest/__init__.py`

- [ ] **Step 1: Delete neutralize_weights_polars and renormalize_weights from utils.py**

In `src/factorium/backtest/utils.py`:

Delete the `neutralize_weights_polars` function (lines 130-159) and the `renormalize_weights` function (lines 215-258). Keep everything else.

- [ ] **Step 2: Update __init__.py exports**

Replace `src/factorium/backtest/__init__.py` with:

```python
from .backtester import (
    IterativeBacktester as LegacyBacktester,
)
from .backtester import (
    IterativeBacktestResult as LegacyBacktestResult,
)
from .allocators import (
    LongOnlyAllocator,
    MarketNeutralAllocator,
    TopNAllocator,
    WeightAllocator,
)
from .constraints import (
    LongOnlyConstraint,
    MarketNeutralConstraint,
    MaxGrossExposureConstraint,
    MaxPositionConstraint,
    WeightConstraint,
)
from .metrics import calculate_metrics
from .normalizers import (
    MinMaxNormalizer,
    Normalizer,
    RankNormalizer,
    RawNormalizer,
    ZScoreNormalizer,
)
from .pipeline import AlphaPipeline
from .portfolio import Portfolio
from .utils import (
    MAX_PERIODS_PER_YEAR,
    MIN_PERIODS_PER_YEAR,
    POSITION_EPSILON,
    frequency_to_periods_per_year,
    neutralize_weights,
    normalize_weights,
    parse_frequency_to_seconds,
)
from .vectorized import BacktestResult, BacktestResultPandas, VectorizedBacktester

Backtester = VectorizedBacktester

__all__ = [
    "AlphaPipeline",
    "Backtester",
    "BacktestResult",
    "BacktestResultPandas",
    "LegacyBacktester",
    "LegacyBacktestResult",
    "LongOnlyAllocator",
    "LongOnlyConstraint",
    "MarketNeutralAllocator",
    "MarketNeutralConstraint",
    "MaxGrossExposureConstraint",
    "MaxPositionConstraint",
    "MAX_PERIODS_PER_YEAR",
    "MIN_PERIODS_PER_YEAR",
    "MinMaxNormalizer",
    "Normalizer",
    "POSITION_EPSILON",
    "Portfolio",
    "RankNormalizer",
    "RawNormalizer",
    "TopNAllocator",
    "VectorizedBacktester",
    "WeightAllocator",
    "WeightConstraint",
    "ZScoreNormalizer",
    "calculate_metrics",
    "frequency_to_periods_per_year",
    "neutralize_weights",
    "normalize_weights",
    "parse_frequency_to_seconds",
]
```

- [ ] **Step 3: Run all backtest tests**

Run: `uv run pytest tests/backtest/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/factorium/backtest/utils.py src/factorium/backtest/__init__.py
git commit -m "refactor(backtest): remove old neutralize/renormalize utils, update exports

Delete neutralize_weights_polars() and renormalize_weights().
Add pipeline, normalizer, and allocator exports to __init__.py."
```

---

### Task 9: Update ResearchSession

**Files:**
- Modify: `src/factorium/research/session.py`

- [ ] **Step 1: Update ResearchSession.backtest() signature**

In `src/factorium/research/session.py`, update the `backtest` method (around line 267):

Replace:

```python
    def backtest(
        self,
        signal: Factor,
        neutralization: str = "market",
        entry_price: str = "close",
        frequency: str | None = None,
        initial_capital: float | None = None,
        transaction_cost: float | None = None,
    ) -> BacktestResult:
```

With:

```python
    def backtest(
        self,
        signal: Factor,
        pipeline: "AlphaPipeline | None" = None,
        entry_price: str = "close",
        frequency: str | None = None,
        initial_capital: float | None = None,
        transaction_cost: float | None = None,
    ) -> BacktestResult:
```

Update the docstring accordingly (replace neutralization description with pipeline description).

Remove the neutralization validation block:

```python
        # Delete these lines:
        if neutralization not in ["market", "none"]:
            raise ValueError(f"neutralization must be 'market' or 'none', got {neutralization}")
```

Update the VectorizedBacktester call:

```python
        bt = VectorizedBacktester(
            prices=self.data,
            signal=signal,
            pipeline=pipeline,
            entry_price=entry_price,
            frequency=frequency or self.default_frequency,
            initial_capital=initial_capital or self.default_initial_capital,
            transaction_cost=transaction_cost or self.default_transaction_cost,
        )
        return bt.run()
```

Add import at top of file (or use TYPE_CHECKING):

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..backtest.pipeline import AlphaPipeline
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS (or identify any remaining failures)

- [ ] **Step 3: Commit**

```bash
git add src/factorium/research/session.py
git commit -m "refactor(research): update ResearchSession.backtest() to accept pipeline

Replace neutralization parameter with pipeline: AlphaPipeline."
```

---

### Task 10: Update documentation

**Files:**
- Modify: `docs/user-guide/backtest.md`

- [ ] **Step 1: Update backtest.md**

Key changes to `docs/user-guide/backtest.md`:

1. **Section 2 (最簡範例)**: Replace `neutralization="market"` with `pipeline=` usage, show both default and custom pipeline examples

2. **Section 3.1 (mask)**: Replace `neutralization="market"` with default pipeline usage

3. **Section 4 (市場中性與 Long-only)**: Replace the neutralization-based explanation with pipeline-based explanation. Show `AlphaPipeline(allocator=MarketNeutralAllocator())` and `AlphaPipeline(allocator=LongOnlyAllocator())`. Remove reference to `backtest.utils.neutralize_weights_polars`.

4. **Section 5 (權重約束)**: Update constraint usage to show constraints inside `AlphaPipeline` instead of as a direct parameter to `Backtester`.

5. **Section 6 (ResearchSession)**: Update `session.backtest()` example to use `pipeline=`.

6. **Add new section**: Brief overview of `AlphaPipeline` three-stage concept, list available Normalizers and Allocators.

- [ ] **Step 2: Verify docs build**

Run: `uv run mkdocs build --strict 2>&1 | head -20`
Expected: Build succeeds without errors

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/backtest.md
git commit -m "docs(backtest): update user guide for pipeline API

Replace neutralization/constraints examples with AlphaPipeline usage."
```

---

### Task 11: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Run type check (if configured)**

Run: `uv run mypy src/factorium/backtest/ --ignore-missing-imports 2>&1 | head -30` or equivalent
Expected: No errors, or state if no type checker is configured

- [ ] **Step 3: Verify no stale references to old API**

```bash
grep -rn "neutralization" src/factorium/backtest/ --include="*.py"
```

Expected: No hits in `vectorized.py` or `utils.py`. The `backtester.py` (legacy IterativeBacktester) will still have it — that's expected since it's a separate deprecated class.

```bash
grep -rn "neutralize_weights_polars\|from .utils import renormalize_weights" src/factorium/ --include="*.py"
```

Expected: No hits.

- [ ] **Step 4: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "chore(backtest): final cleanup for alpha pipeline migration"
```
