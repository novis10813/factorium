"""
Parquet storage utilities with Hive partitioning support.

Provides functions for CSV to Parquet conversion and optimized reading via DuckDB.
"""

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import duckdb
from pathlib import Path
from typing import Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Column definitions for Binance data types (for headerless CSVs)
BINANCE_COLUMNS = {
    'aggTrades': [
        'agg_trade_id', 'price', 'quantity', 'first_trade_id',
        'last_trade_id', 'transact_time', 'is_buyer_maker'
    ],
    'trades': [
        'id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'
    ],
    'klines': [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ],
}


def _detect_has_header(csv_path: Path) -> bool:
    """
    Detect if CSV has a header by checking if first row contains numeric values.
    Binance data without headers will have numeric first column (trade id).
    """
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
        if not first_line:
            return False
        first_value = first_line.split(',')[0]
        # If first value is numeric, it's likely data (no header)
        try:
            float(first_value)
            return False
        except ValueError:
            return True


def csv_to_parquet(
    csv_path: Path,
    output_dir: Path,
    compression: str = 'zstd',
    filename: str = 'data.parquet',
    data_type: Optional[str] = None,
) -> Path:
    """
    Convert CSV file to Parquet format in target directory.
    
    Automatically detects if CSV has headers. For headerless Binance CSVs,
    uses predefined column names based on data_type.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to write Parquet file (will be created if needed)
        compression: Compression codec ('zstd', 'snappy', 'gzip', or None)
        filename: Output filename
        data_type: Binance data type (aggTrades, trades, klines) for headerless CSVs
        
    Returns:
        Path to created Parquet file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    has_header = _detect_has_header(csv_path)
    
    if has_header:
        # CSV has headers, read normally
        table = pv.read_csv(csv_path)
    else:
        # No header - use predefined column names
        if data_type and data_type in BINANCE_COLUMNS:
            column_names = BINANCE_COLUMNS[data_type]
            read_options = pv.ReadOptions(column_names=column_names)
            table = pv.read_csv(csv_path, read_options=read_options)
            logger.debug(f"Applied column names for {data_type}: {column_names}")
        else:
            # Fallback: read with auto-generated column names
            logger.warning(f"No column definitions for data_type={data_type}, using auto-generated names")
            table = pv.read_csv(csv_path)
    
    out_path = output_dir / filename
    pq.write_table(table, out_path, compression=compression)
    logger.debug(f"Converted {csv_path} -> {out_path}")
    return out_path


def read_hive_parquet(
    base_path: str,
    columns: Optional[List[str]] = None,
    where: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read Parquet files with Hive partitioning via DuckDB.
    
    Args:
        base_path: Glob pattern to Parquet files (e.g., 'Data/market=*/**/*.parquet')
        columns: Optional list of columns to select
        where: Optional WHERE clause (without 'WHERE' keyword)
        
    Returns:
        DataFrame with query results
        
    Example:
        >>> df = read_hive_parquet(
        ...     'Data/market=futures_um/data_type=klines/**/*.parquet',
        ...     columns=['open', 'high', 'low', 'close', 'volume'],
        ...     where="symbol = 'BTCUSDT' AND year = 2024"
        ... )
    """
    col_str = ", ".join(columns) if columns else "*"
    query = f"""
        SELECT {col_str}
        FROM read_parquet('{base_path}', hive_partitioning=true)
        {f'WHERE {where}' if where else ''}
    """
    return duckdb.query(query).df()


def build_hive_path(
    base_path: Path,
    market: str,
    data_type: str,
    symbol: str,
    year: int,
    month: int,
    day: int,
) -> Path:
    """
    Build Hive-style partition path.
    
    Args:
        base_path: Base data directory
        market: Market type (futures_cm, futures_um, spot)
        data_type: Data type (trades, klines, aggTrades)
        symbol: Trading symbol
        year: Year
        month: Month (1-12)
        day: Day (1-31)
        
    Returns:
        Path to partition directory
    """
    return (
        base_path
        / f"market={market}"
        / f"data_type={data_type}"
        / f"symbol={symbol}"
        / f"year={year}"
        / f"month={month:02d}"
        / f"day={day:02d}"
    )


def get_market_string(market_type: str, futures_type: str = '') -> str:
    """
    Get combined market string for Hive partition.
    
    Args:
        market_type: 'spot' or 'futures'
        futures_type: 'cm' or 'um' (only for futures)
        
    Returns:
        Market string: 'spot', 'futures_cm', or 'futures_um'
    """
    if market_type == 'spot':
        return 'spot'
    return f"{market_type}_{futures_type}"
