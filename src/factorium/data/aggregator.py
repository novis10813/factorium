"""DuckDB-based bar aggregator for high-performance OHLCV aggregation."""

import logging

import duckdb
import polars as pl

from .adapters.base import ColumnMapping

logger = logging.getLogger(__name__)


class BarAggregator:
    """High-performance bar aggregator using DuckDB SQL.

    Aggregates tick/trade data into OHLCV bars directly in DuckDB,
    avoiding loading raw tick data into Python memory.

    Benefits:
    - Memory efficient: aggregation happens in DuckDB
    - Fast: uses DuckDB's vectorized execution engine
    - Scalable: handles large datasets that don't fit in memory
    """

    def _compute_metadata(self, df: pl.DataFrame) -> "AggBarMetadata":
        """Compute metadata from aggregated DataFrame.

        Args:
            df: Polars DataFrame with aggregated data

        Returns:
            AggBarMetadata with symbols, time range, and row count
        """
        from .metadata import AggBarMetadata

        if df.is_empty():
            return AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)

        symbols = df.select(pl.col("symbol").unique().sort()).to_series().to_list()
        min_time = df["start_time"].min()
        max_time = df["end_time"].max()
        num_rows = df.height

        return AggBarMetadata(
            symbols=symbols,
            min_time=min_time,
            max_time=max_time,
            num_rows=num_rows,
        )

    def _get_buyer_seller_sql(self, include: bool, ibm_col: str | None) -> tuple[str, str]:
        """Generate buyer/seller statistics SQL fragments.

        Args:
            include: Whether to include buyer/seller stats
            ibm_col: Column name for is_buyer_maker, or None if not available

        Returns:
            Tuple of (aggregation_sql, select_cols) for buyer/seller stats.
            Both empty strings if not included.
        """
        if not include or not ibm_col:
            return "", ""

        agg_sql = """
            , SUM(CASE WHEN NOT is_buyer_maker THEN 1 ELSE 0 END) AS num_buyer
            , SUM(CASE WHEN is_buyer_maker THEN 1 ELSE 0 END) AS num_seller
            , SUM(CASE WHEN NOT is_buyer_maker THEN volume ELSE 0 END) AS num_buyer_volume
            , SUM(CASE WHEN is_buyer_maker THEN volume ELSE 0 END) AS num_seller_volume
        """
        cols = ", num_buyer, num_seller, num_buyer_volume, num_seller_volume"
        return agg_sql, cols

    def aggregate_time_bars(
        self,
        parquet_pattern: str,
        symbols: list[str],
        interval_ms: int,
        start_ts: int,
        end_ts: int,
        column_mapping: ColumnMapping,
        include_buyer_seller: bool = True,
    ) -> tuple[pl.DataFrame, "AggBarMetadata"]:
        """Aggregate tick data into time-based OHLCV bars.

        Args:
            parquet_pattern: Glob pattern for Parquet files
            symbols: List of symbols to include
            interval_ms: Bar interval in milliseconds
            start_ts: Start timestamp (milliseconds)
            end_ts: End timestamp (milliseconds)
            column_mapping: Column name mapping for the data source
            include_buyer_seller: Include buyer/seller statistics

        Returns:
            Tuple of (Polars DataFrame, AggBarMetadata) with columns:
            - symbol, start_time, end_time
            - open, high, low, close, volume, vwap
            - (optional) num_buyer, num_seller, num_buyer_volume, num_seller_volume
        """
        ts_col = column_mapping.timestamp
        price_col = column_mapping.price
        volume_col = column_mapping.volume
        ibm_col = column_mapping.is_buyer_maker

        # SQL injection prevention / quoting safety for symbol list
        escaped_symbols = [s.replace("'", "''") for s in symbols]
        symbols_str = ", ".join([f"'{s}'" for s in escaped_symbols])

        buyer_seller_sql, buyer_seller_cols = self._get_buyer_seller_sql(include_buyer_seller, ibm_col)

        query = f"""
            WITH raw_data AS (
                SELECT
                    symbol,
                    {ts_col} AS ts,
                    {price_col} AS price,
                    {volume_col} AS volume
                    {f", {ibm_col} AS is_buyer_maker" if include_buyer_seller and ibm_col else ""}
                FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
                WHERE symbol IN ({symbols_str})
                  AND {ts_col} >= {start_ts}
                  AND {ts_col} < {end_ts}
            ),
            numbered AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol
                        ORDER BY ts, price, volume{", is_buyer_maker" if include_buyer_seller and ibm_col else ""}
                    ) AS seq
                FROM raw_data
            ),
            aggregated AS (
                SELECT
                    symbol,
                    (ts // {interval_ms}) * {interval_ms} AS start_time,
                    (ts // {interval_ms}) * {interval_ms} + {interval_ms} AS end_time,
                    ARG_MIN(price, seq) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    ARG_MAX(price, seq) AS close,
                    SUM(volume) AS volume,
                    CASE
                        WHEN SUM(volume) <= 1e-10 THEN NULL
                        ELSE SUM(price * volume) / SUM(volume)
                    END AS vwap
                    {buyer_seller_sql}
                FROM numbered
                GROUP BY symbol, (ts // {interval_ms})
            )
            SELECT
                symbol, start_time, end_time,
                open, high, low, close, volume, vwap
                {buyer_seller_cols}
            FROM aggregated
            ORDER BY start_time, symbol
        """

        try:
            with duckdb.connect() as conn:
                df = conn.execute(query).pl()
            metadata = self._compute_metadata(df)
            return df, metadata
        except duckdb.IOException as e:
            from .metadata import AggBarMetadata

            logger.warning(f"DuckDB aggregation failed for pattern '{parquet_pattern}': {e}")
            return pl.DataFrame(), AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)

    def aggregate_tick_bars(
        self,
        parquet_pattern: str,
        symbol: str,
        interval_ticks: int,
        column_mapping: ColumnMapping,
        include_buyer_seller: bool = True,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> tuple[pl.DataFrame, "AggBarMetadata"]:
        """Aggregate tick data into tick-based OHLCV bars.

        Args:
            parquet_pattern: Glob pattern for Parquet files
            symbol: Single symbol (tick bars don't align across symbols)
            interval_ticks: Number of ticks per bar
            column_mapping: Column name mapping for the data source
            include_buyer_seller: Include buyer/seller statistics
            start_ts: Start timestamp in milliseconds (optional)
            end_ts: End timestamp in milliseconds (optional)

        Returns:
            Tuple of (Polars DataFrame, AggBarMetadata) with columns:
            - symbol, start_time, end_time
            - open, high, low, close, volume, vwap
            - (optional) num_buyer, num_seller, num_buyer_volume, num_seller_volume
        """
        if interval_ticks <= 0:
            raise ValueError(f"interval_ticks must be positive, got {interval_ticks}")

        ts_col = column_mapping.timestamp
        price_col = column_mapping.price
        volume_col = column_mapping.volume
        ibm_col = column_mapping.is_buyer_maker

        # SQL injection prevention
        escaped_symbol = symbol.replace("'", "''")

        # Build time filter
        time_filter = ""
        if start_ts is not None:
            time_filter += f" AND {ts_col} >= {start_ts}"
        if end_ts is not None:
            time_filter += f" AND {ts_col} < {end_ts}"

        buyer_seller_sql, buyer_seller_cols = self._get_buyer_seller_sql(include_buyer_seller, ibm_col)

        query = f"""
            WITH raw_data AS (
                SELECT
                    symbol,
                    {ts_col} AS ts,
                    {price_col} AS price,
                    {volume_col} AS volume
                    {f", {ibm_col} AS is_buyer_maker" if include_buyer_seller and ibm_col else ""}
                FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
                WHERE symbol = '{escaped_symbol}'{time_filter}
            ),
            numbered AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        ORDER BY ts, price, volume{", is_buyer_maker" if include_buyer_seller and ibm_col else ""}
                    ) AS seq
                FROM raw_data
            ),
            with_bar_id AS (
                SELECT
                    *,
                    (seq - 1) // {interval_ticks} AS bar_id
                FROM numbered
            ),
            aggregated AS (
                SELECT
                    symbol,
                    bar_id,
                    ARG_MIN(ts, seq) AS start_time,
                    ARG_MAX(ts, seq) AS end_time,
                    ARG_MIN(price, seq) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    ARG_MAX(price, seq) AS close,
                    SUM(volume) AS volume,
                    CASE
                        WHEN SUM(volume) <= 1e-10 THEN NULL
                        ELSE SUM(price * volume) / SUM(volume)
                    END AS vwap
                    {buyer_seller_sql}
                FROM with_bar_id
                GROUP BY symbol, bar_id
            )
            SELECT
                symbol, start_time, end_time,
                open, high, low, close, volume, vwap
                {buyer_seller_cols}
            FROM aggregated
            ORDER BY bar_id
        """

        try:
            with duckdb.connect() as conn:
                df = conn.execute(query).pl()
            metadata = self._compute_metadata(df)
            return df, metadata
        except duckdb.IOException as e:
            from .metadata import AggBarMetadata

            logger.warning(f"DuckDB tick bar aggregation failed for pattern '{parquet_pattern}': {e}")
            return pl.DataFrame(), AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)

    def aggregate_volume_bars(
        self,
        parquet_pattern: str,
        symbol: str,
        interval_volume: float,
        column_mapping: ColumnMapping,
        include_buyer_seller: bool = True,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> tuple[pl.DataFrame, "AggBarMetadata"]:
        """Aggregate tick data into volume-based OHLCV bars.

        Bar boundary rule (Greedy Packing):
        - Accumulate volume until cumulative >= threshold
        - When threshold crossed, the current trade belongs to current bar
        - Reset accumulator and next trade starts a new bar

        Implemented using Recursive CTE for correctness.

        Args:
            parquet_pattern: Glob pattern for Parquet files
            symbol: Single symbol (volume bars don't align across symbols)
            interval_volume: Volume threshold per bar
            column_mapping: Column name mapping for the data source
            include_buyer_seller: Include buyer/seller statistics
            start_ts: Start timestamp in milliseconds (optional)
            end_ts: End timestamp in milliseconds (optional)

        Returns:
            Tuple of (Polars DataFrame, AggBarMetadata) with columns:
            - symbol, start_time, end_time
            - open, high, low, close, volume, vwap
            - (optional) num_buyer, num_seller, num_buyer_volume, num_seller_volume
        """
        if interval_volume <= 0:
            raise ValueError(f"interval_volume must be positive, got {interval_volume}")

        ts_col = column_mapping.timestamp
        price_col = column_mapping.price
        volume_col = column_mapping.volume
        ibm_col = column_mapping.is_buyer_maker

        # SQL injection prevention
        escaped_symbol = symbol.replace("'", "''")

        # Build time filter
        time_filter = ""
        if start_ts is not None:
            time_filter += f" AND {ts_col} >= {start_ts}"
        if end_ts is not None:
            time_filter += f" AND {ts_col} < {end_ts}"

        buyer_seller_sql, buyer_seller_cols = self._get_buyer_seller_sql(include_buyer_seller, ibm_col)

        query = f"""
            WITH RECURSIVE
            raw_data AS (
                SELECT
                    symbol,
                    {ts_col} AS ts,
                    {price_col} AS price,
                    {volume_col} AS volume
                    {f", {ibm_col} AS is_buyer_maker" if include_buyer_seller and ibm_col else ""}
                FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
                WHERE symbol = '{escaped_symbol}'{time_filter}
            ),
            numbered AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        ORDER BY ts, price, volume{", is_buyer_maker" if include_buyer_seller and ibm_col else ""}
                    ) AS seq
                FROM raw_data
            ),
            -- Recursive CTE to compute greedy bar assignments (Greedy Packing algorithm)
            greedy AS (
                -- Base case: first row
                SELECT 
                    seq,
                    volume,
                    volume AS running_volume,
                    CAST(0 AS BIGINT) AS bar_id
                FROM numbered
                WHERE seq = 1
                
                UNION ALL
                
                -- Recursive case: process remaining rows
                SELECT
                    r.seq,
                    r.volume,
                    CASE 
                        WHEN g.running_volume >= {interval_volume} THEN r.volume  -- Reset after threshold
                        ELSE g.running_volume + r.volume                          -- Continue accumulating
                    END AS running_volume,
                    CASE 
                        WHEN g.running_volume >= {interval_volume} THEN g.bar_id + 1  -- New bar
                        ELSE g.bar_id                                                  -- Same bar
                    END AS bar_id
                FROM numbered r
                JOIN greedy g ON r.seq = g.seq + 1
            ),
            with_bar_id AS (
                SELECT n.*, g.bar_id
                FROM numbered n
                JOIN greedy g ON n.seq = g.seq
            ),
            aggregated AS (
                SELECT
                    symbol,
                    bar_id,
                    ARG_MIN(ts, seq) AS start_time,
                    ARG_MAX(ts, seq) AS end_time,
                    ARG_MIN(price, seq) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    ARG_MAX(price, seq) AS close,
                    SUM(volume) AS volume,
                    CASE
                        WHEN SUM(volume) <= 1e-10 THEN NULL
                        ELSE SUM(price * volume) / SUM(volume)
                    END AS vwap
                    {buyer_seller_sql}
                FROM with_bar_id
                GROUP BY symbol, bar_id
            )
            SELECT
                symbol, start_time, end_time,
                open, high, low, close, volume, vwap
                {buyer_seller_cols}
            FROM aggregated
            ORDER BY bar_id
        """

        try:
            with duckdb.connect() as conn:
                df = conn.execute(query).pl()
            metadata = self._compute_metadata(df)
            return df, metadata
        except duckdb.IOException as e:
            from .metadata import AggBarMetadata

            logger.warning(f"DuckDB volume bar aggregation failed for pattern '{parquet_pattern}': {e}")
            return pl.DataFrame(), AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)

    def aggregate_dollar_bars(
        self,
        parquet_pattern: str,
        symbol: str,
        interval_dollar: float,
        column_mapping: ColumnMapping,
        include_buyer_seller: bool = True,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> tuple[pl.DataFrame, "AggBarMetadata"]:
        """Aggregate tick data into dollar-volume based OHLCV bars.

        Bar boundary rule: Same as volume bars, but threshold is dollar volume.
        When cumulative dollar volume (price * volume) >= threshold, the current
        trade belongs to the current bar, and the next trade starts a new bar.

        Args:
            parquet_pattern: Glob pattern for Parquet files
            symbol: Single symbol (dollar bars don't align across symbols)
            interval_dollar: Dollar volume threshold per bar
            column_mapping: Column name mapping for the data source
            include_buyer_seller: Include buyer/seller statistics
            start_ts: Start timestamp in milliseconds (optional)
            end_ts: End timestamp in milliseconds (optional)

        Returns:
            Tuple of (Polars DataFrame, AggBarMetadata) with columns:
            - symbol, start_time, end_time
            - open, high, low, close, volume, vwap
            - (optional) num_buyer, num_seller, num_buyer_volume, num_seller_volume
        """
        if interval_dollar <= 0:
            raise ValueError(f"interval_dollar must be positive, got {interval_dollar}")

        ts_col = column_mapping.timestamp
        price_col = column_mapping.price
        volume_col = column_mapping.volume
        ibm_col = column_mapping.is_buyer_maker

        # SQL injection prevention
        escaped_symbol = symbol.replace("'", "''")

        # Build time filter
        time_filter = ""
        if start_ts is not None:
            time_filter += f" AND {ts_col} >= {start_ts}"
        if end_ts is not None:
            time_filter += f" AND {ts_col} < {end_ts}"

        buyer_seller_sql, buyer_seller_cols = self._get_buyer_seller_sql(include_buyer_seller, ibm_col)

        query = f"""
            WITH RECURSIVE
            raw_data AS (
                SELECT
                    symbol,
                    {ts_col} AS ts,
                    {price_col} AS price,
                    {volume_col} AS volume,
                    {price_col} * {volume_col} AS dollar_volume
                    {f", {ibm_col} AS is_buyer_maker" if include_buyer_seller and ibm_col else ""}
                FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
                WHERE symbol = '{escaped_symbol}'{time_filter}
            ),
            numbered AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        ORDER BY ts, price, volume{", is_buyer_maker" if include_buyer_seller and ibm_col else ""}
                    ) AS seq
                FROM raw_data
            ),
            -- Recursive CTE to compute greedy bar assignments (Greedy Packing algorithm)
            greedy AS (
                -- Base case: first row
                SELECT 
                    seq,
                    dollar_volume,
                    dollar_volume AS running_dollar,
                    CAST(0 AS BIGINT) AS bar_id
                FROM numbered
                WHERE seq = 1
                
                UNION ALL
                
                -- Recursive case: process remaining rows
                SELECT
                    r.seq,
                    r.dollar_volume,
                    CASE 
                        WHEN g.running_dollar >= {interval_dollar} THEN r.dollar_volume  -- Reset after threshold
                        ELSE g.running_dollar + r.dollar_volume                          -- Continue accumulating
                    END AS running_dollar,
                    CASE 
                        WHEN g.running_dollar >= {interval_dollar} THEN g.bar_id + 1  -- New bar
                        ELSE g.bar_id                                                  -- Same bar
                    END AS bar_id
                FROM numbered r
                JOIN greedy g ON r.seq = g.seq + 1
            ),
            with_bar_id AS (
                SELECT n.*, g.bar_id
                FROM numbered n
                JOIN greedy g ON n.seq = g.seq
            ),
            aggregated AS (
                SELECT
                    symbol,
                    bar_id,
                    ARG_MIN(ts, seq) AS start_time,
                    ARG_MAX(ts, seq) AS end_time,
                    ARG_MIN(price, seq) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    ARG_MAX(price, seq) AS close,
                    SUM(volume) AS volume,
                    CASE
                        WHEN SUM(volume) <= 1e-10 THEN NULL
                        ELSE SUM(price * volume) / SUM(volume)
                    END AS vwap
                    {buyer_seller_sql}
                FROM with_bar_id
                GROUP BY symbol, bar_id
            )
            SELECT
                symbol, start_time, end_time,
                open, high, low, close, volume, vwap
                {buyer_seller_cols}
            FROM aggregated
            ORDER BY bar_id
        """

        try:
            with duckdb.connect() as conn:
                df = conn.execute(query).pl()
            metadata = self._compute_metadata(df)
            return df, metadata
        except duckdb.IOException as e:
            from .metadata import AggBarMetadata

            logger.warning(f"DuckDB dollar bar aggregation failed for pattern '{parquet_pattern}': {e}")
            return pl.DataFrame(), AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)
