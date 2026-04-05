# Plan: Polars Expression Engine & Evolutionary Architecture

**Status**: Draft
**Created**: 2026-01-26
**Based on**: `Factorium Evolution Idea.md`

## 1. Vision
Transform Factorium from a manual factor research tool into an automated Alpha mining system. The core engine will migrate from eager Pandas execution to lazy **Polars Expressions**, enabling high-performance backtesting and serving as the "DNA" for an evolutionary algorithm.

## 2. Motivation
*   **Performance**: Polars' Lazy API and automatic query optimization are significantly faster than Pandas, especially for large datasets.
*   **Mutability**: Expression trees are easier to programmatically manipulate (mutate/crossover) than string-based formulas or imperative code.
*   **Serialization**: Expression trees can be serialized to JSON/Binary, allowing efficient storage in a Knowledge Graph.

## 3. Architecture Overview

### 3.1 The "Factor DNA" (Polars Expression)
Instead of holding `np.array` or `pd.Series` directly, a `Factor` will hold a `pl.Expr`.
*   **Old**: `factor.data` (DataFrame)
*   **New**: `factor.expr` (Expression Tree) + `factor.context` (LazyFrame)

Example:
```python
# A factor is just a recipe, not the data itself
f = (pl.col("close") - pl.col("open")) / pl.col("open")
```

### 3.2 Knowledge Graph (Metadata Layer)
Stores the structure and performance of factors.
*   **Nodes**: Operators (`Mean`, `Sum`, `Rank`), Primitives (`Open`, `Vol`), Constants.
*   **Edges**: Data flow.
*   **Metrics**: Sharpe, Sortino, IC, Correlation with existing factors.

### 3.3 Evolutionary Engine
*   **Framework**: Python `deap` or custom loop.
*   **Operations**:
    *   **Mutation**: Change window size (`mean(10)` -> `mean(20)`), swap operators (`max` -> `min`).
    *   **Crossover**: Swap sub-trees between two parent factors.
*   **Fitness Function**: Vectorized backtest on validation set.

## 4. Implementation Roadmap

### Phase 1: Polars Integration (The Foundation)
*   [x] Introduce `polars` dependency.
*   [x] Create `PolarsDataLoader`: Load Parquet data directly into `pl.LazyFrame`. *(or equivalent Polars-based loader in `factorium.data` stack)*
*   [x] Prototype `PolarsFactor`: A wrapper class that builds `pl.Expr`.
    *   [x] Implement basic operators: `ts_mean`, `ts_rank`, `cs_rank` on top of Polars.

### Phase 2: Expression Tree Manipulation
*   [x] Implement "Expression Walker": A utility to traverse and modify the Polars expression tree. *(can be a custom AST that compiles to `pl.Expr`, not necessarily Polars’ internal AST)*
    *   *Challenge*: Polars internal AST isn't fully exposed. May need a shadow tree or custom AST builder that compiles *to* Polars Expr.
*   [x] Serialize/Deserialize factors to JSON (or an equivalent structured representation suitable for future storage).

### Phase 3: The Evolutionary Loop
*   [ ] Setup `EvoEngine` class.
*   [ ] Implement `generate_random_factor()`.
*   [ ] Implement `mutate(factor)` and `crossover(factor_a, factor_b)`.
*   [ ] Build a vectorized backtester using Polars (`metrics.py` adaptation).

### Phase 4: Knowledge Graph & Storage
*   [ ] Design Schema (Nodes/Edges).
*   [ ] Implement storage interface (SQLite/Neo4j or simple NetworkX dump).
*   [ ] Connect Engine to MinIO for data persistence.

## 5. Technical Challenges
*   **Polars Introspection**: Modifying a `pl.Expr` after creation is hard.
    *   *Solution*: We likely need our own lightweight AST (Abstract Syntax Tree) classes (`Add`, `RollingMean`, `Feature`) that *render* to `pl.Expr`. The genetic alg modifies our AST, not the Polars object directly.
*   **Data Alignment**: Ensuring cross-sectional ops (`cs_rank`) work correctly on LazyFrames without materializing too early.

