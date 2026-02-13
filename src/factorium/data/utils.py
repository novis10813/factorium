"""
Shared utilities for data loading and processing.
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def calculate_date_range(
    start_date: str | None = None,
    end_date: str | None = None,
    days: int | None = None,
    default_days: int = 7,
) -> tuple[datetime, datetime]:
    """Calculate date range aligned to UTC midnight.

    All returned datetimes are snapped to midnight (00:00:00) to ensure
    daily processing boundaries align with bar interval buckets.
    Without this alignment, a boundary like 09:32:22 UTC would split
    the 09:32 minute-bucket across two adjacent daily queries,
    producing duplicate bars with partial OHLCV data.

    Priority:
    1. If both start_date and end_date are provided: [start, end]
    2. If start_date and days are provided: [start, start + days]
    3. If neither: [today_midnight - default_days, today_midnight + 1]
    4. If only days: [today_midnight - days, today_midnight + 1]

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        days: Number of days
        default_days: Default number of days if none specified (default: 7)

    Returns:
        tuple[datetime, datetime]: (start_dt, end_dt) snapped to midnight.
    """
    try:
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if start > end:
                raise ValueError("Start date must be earlier than or equal to end date")
            return start, end

        if start_date and days:
            if days < 1:
                raise ValueError("Days must be greater than 0")
            start = datetime.strptime(start_date, "%Y-%m-%d")
            return start, start + timedelta(days=days)

        # Snap to UTC midnight for consistent daily boundaries
        today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        # end = start of tomorrow (exclusive) to include today's full data
        end = today_midnight + timedelta(days=1)

        if days:
            if days < 1:
                raise ValueError("Days must be greater than 0")
            start = end - timedelta(days=days)
        else:
            start = end - timedelta(days=default_days)

        return start, end

    except Exception as e:
        logger.error(f"Error calculating date range: {str(e)}")
        raise
