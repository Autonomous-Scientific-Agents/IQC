#!/usr/bin/env python3
"""Inspect parquet columns and optionally drop columns by index."""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ColumnStats:
    """Basic stats for a single parquet column."""

    index: int
    name: str
    data_type: str
    null_count: int
    null_pct: float
    compressed_bytes: int
    uncompressed_bytes: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Read a parquet file, list columns with basic stats, and optionally "
            "write a new parquet file with selected columns removed."
        )
    )
    parser.add_argument("input_file", help="Path to the input parquet file")
    parser.add_argument(
        "--drop",
        nargs="+",
        metavar="IDX",
        help=(
            "Column indices to drop. Accepts space-separated values like "
            "'--drop 1 4 7' or comma-separated values like '--drop 1,4,7'."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output path for the reduced parquet file. Must not exist.",
    )
    return parser.parse_args(argv)


def parse_drop_indices(raw_values: Optional[Sequence[str]]) -> List[int]:
    """Parse drop indices from comma- or space-separated CLI values."""
    if not raw_values:
        return []

    indices: List[int] = []
    for raw_value in raw_values:
        for item in raw_value.split(","):
            value = item.strip()
            if not value:
                continue
            try:
                index = int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid column index '{value}'.") from exc
            if index < 0:
                raise ValueError(f"Column indices must be non-negative: {index}")
            indices.append(index)

    if not indices:
        raise ValueError("No valid column indices were provided to --drop.")

    return sorted(dict.fromkeys(indices))


def format_bytes(size: int) -> str:
    """Format a byte count using a compact human-readable unit."""
    sign = "-" if size < 0 else ""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(abs(size))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{sign}{int(value)} {unit}"
            return f"{sign}{value:.1f} {unit}"
        value /= 1024
    return f"{sign}{abs(size)} B"


def truncate(text: str, width: int) -> str:
    """Truncate long strings for compact tabular output."""
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def infer_output_path(input_path: Path) -> Path:
    """Infer a non-conflicting output path next to the input file."""
    base_name = input_path.stem
    candidate = input_path.with_name(f"{base_name}_reduced.parquet")
    counter = 1

    while candidate.exists() or candidate.resolve() == input_path.resolve():
        candidate = input_path.with_name(f"{base_name}_reduced_{counter}.parquet")
        counter += 1

    return candidate


def resolve_output_path(input_path: Path, output_file: Optional[str]) -> Path:
    """Resolve the output path without overwriting existing files."""
    if output_file is None:
        return infer_output_path(input_path)

    output_path = Path(output_file)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path must be different from the input file.")
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")
    return output_path


def collect_storage_sizes(parquet_file: pq.ParquetFile) -> Dict[str, Tuple[int, int]]:
    """Collect compressed and uncompressed sizes per top-level column."""
    sizes: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    metadata = parquet_file.metadata

    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            column_name = column.path_in_schema.split(".")[0]
            sizes[column_name][0] += max(column.total_compressed_size, 0)
            sizes[column_name][1] += max(column.total_uncompressed_size, 0)

    return {name: (values[0], values[1]) for name, values in sizes.items()}


def collect_column_stats(
    table: pa.Table, parquet_file: pq.ParquetFile
) -> List[ColumnStats]:
    """Build column stats in the table's original column order."""
    storage_sizes = collect_storage_sizes(parquet_file)
    num_rows = len(table)
    stats: List[ColumnStats] = []

    for index, field in enumerate(table.schema):
        column = table[field.name]
        null_count = column.null_count
        null_pct = (null_count / num_rows * 100) if num_rows else 0.0
        compressed_bytes, uncompressed_bytes = storage_sizes.get(field.name, (0, 0))
        stats.append(
            ColumnStats(
                index=index,
                name=field.name,
                data_type=str(field.type),
                null_count=null_count,
                null_pct=null_pct,
                compressed_bytes=compressed_bytes,
                uncompressed_bytes=uncompressed_bytes,
            )
        )

    return stats


def print_column_report(
    input_path: Path, parquet_file: pq.ParquetFile, stats: Iterable[ColumnStats]
) -> None:
    """Print file info plus a compact per-column summary."""
    stats = list(stats)
    total_compressed = sum(item.compressed_bytes for item in stats)

    print(f"File    : {input_path}")
    print(f"Rows    : {parquet_file.metadata.num_rows:,}")
    print(f"Columns : {len(stats)}")
    print(f"Size    : {format_bytes(input_path.stat().st_size)}")
    print()
    print("Columns:")
    header = (
        f"{'Idx':>3}  {'Name':<28}  {'Type':<18}  "
        f"{'Nulls':>10}  {'Null %':>7}  {'Comp.':>10}  {'Share':>7}"
    )
    print(header)
    print("-" * len(header))

    for item in stats:
        share_pct = (
            item.compressed_bytes / total_compressed * 100 if total_compressed else 0.0
        )
        print(
            f"{item.index:>3}  "
            f"{truncate(item.name, 28):<28}  "
            f"{truncate(item.data_type, 18):<18}  "
            f"{item.null_count:>10,}  "
            f"{item.null_pct:>6.1f}%  "
            f"{format_bytes(item.compressed_bytes):>10}  "
            f"{share_pct:>6.1f}%"
        )

    print()
    print("Use --drop IDX [IDX ...] to write a reduced parquet file.")


def validate_drop_indices(drop_indices: Sequence[int], num_columns: int) -> None:
    """Validate requested drop indices against the table schema."""
    invalid = [index for index in drop_indices if index >= num_columns]
    if invalid:
        formatted = ", ".join(str(index) for index in invalid)
        raise IndexError(f"Column index out of range: {formatted}")

    if len(drop_indices) >= num_columns:
        raise ValueError("Refusing to drop every column from the parquet file.")


def write_reduced_parquet(
    table: pa.Table, output_path: Path, keep_columns: Sequence[str]
) -> None:
    """Write a reduced parquet file with size-focused defaults."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced_table = table.select(list(keep_columns))
    pq.write_table(
        reduced_table,
        output_path,
        compression="zstd",
        use_dictionary=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the parquet inspection/drop CLI."""
    args = parse_args(argv)
    input_path = Path(args.input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    if args.output and not args.drop:
        raise ValueError("--output can only be used together with --drop.")

    parquet_file = pq.ParquetFile(input_path)
    table = parquet_file.read()
    stats = collect_column_stats(table, parquet_file)

    drop_indices = parse_drop_indices(args.drop)
    if not drop_indices:
        print_column_report(input_path, parquet_file, stats)
        return 0

    validate_drop_indices(drop_indices, len(stats))
    output_path = resolve_output_path(input_path, args.output)
    drop_set = set(drop_indices)
    dropped_columns = [item for item in stats if item.index in drop_set]
    keep_columns = [item.name for item in stats if item.index not in drop_set]

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print("Dropping columns:")
    for item in dropped_columns:
        print(f"  [{item.index}] {item.name} ({item.data_type})")

    write_reduced_parquet(table, output_path, keep_columns)

    original_size = input_path.stat().st_size
    new_size = output_path.stat().st_size
    savings = original_size - new_size
    savings_pct = (savings / original_size * 100) if original_size else 0.0

    print()
    print(f"Kept columns : {len(keep_columns)} / {len(stats)}")
    print(f"Original size: {format_bytes(original_size)}")
    print(f"Reduced size : {format_bytes(new_size)}")
    print(f"Size delta   : {format_bytes(savings)} ({savings_pct:.1f}%)")

    return 0


def run_cli() -> int:
    """CLI wrapper with consistent user-facing error handling."""
    try:
        return main()
    except (FileExistsError, FileNotFoundError, IndexError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
