"""Metadata container for AggBar."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AggBarMetadata:
    """Immutable metadata for AggBar.

    Stores summary statistics computed from actual data,
    not from loader parameters.

    Attributes:
        symbols: List of unique symbols in the data
        min_time: Minimum start_time in milliseconds
        max_time: Maximum end_time in milliseconds
        num_rows: Total number of rows
    """

    symbols: list[str]
    min_time: int
    max_time: int
    num_rows: int

    @classmethod
    def merge(cls, metadata_list: list["AggBarMetadata"]) -> "AggBarMetadata":
        """Merge multiple metadata objects into one.

        Args:
            metadata_list: List of metadata objects to merge.

        Returns:
            Merged metadata with combined symbols, min/max time range, and row count.

        Raises:
            ValueError: If metadata_list is empty.
        """
        if not metadata_list:
            raise ValueError("Cannot merge empty metadata list")

        if len(metadata_list) == 1:
            m = metadata_list[0]
            return cls(
                symbols=list(m.symbols),
                min_time=m.min_time,
                max_time=m.max_time,
                num_rows=m.num_rows,
            )

        # Collect all unique symbols (preserve order, deduplicate)
        seen = set()
        all_symbols = []
        for m in metadata_list:
            for s in m.symbols:
                if s not in seen:
                    seen.add(s)
                    all_symbols.append(s)

        return cls(
            symbols=all_symbols,
            min_time=min(m.min_time for m in metadata_list),
            max_time=max(m.max_time for m in metadata_list),
            num_rows=sum(m.num_rows for m in metadata_list),
        )
