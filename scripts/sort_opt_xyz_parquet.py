#!/usr/bin/env python3
"""Sort a parquet file by atom count and keep selected structure data."""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SORT_COLUMN = "number_of_atoms"
TIE_BREAK_COLUMN = "number_of_electrons"
OUTPUT_COLUMNS = [
    "xyz_file",
    SORT_COLUMN,
    TIE_BREAK_COLUMN,
    "formula",
    "unique_name",
    "opt_xyz",
]


@dataclass(frozen=True)
class DistributionStats:
    """Summary of the number_of_atoms distribution in the input."""

    total_rows: int
    null_rows: int
    min_atoms: Optional[float]
    max_atoms: Optional[float]
    mean_atoms: Optional[float]
    counts: List[Tuple[Any, int]]


@dataclass(frozen=True)
class SortResult:
    """Result metadata for a sorted parquet write."""

    output_path: Path
    stats: DistributionStats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Read a parquet file, sort rows by number_of_atoms from largest to "
            "smallest by default, break atom-count ties by larger "
            "number_of_electrons first, and write a new parquet file containing "
            "only selected structure columns."
        )
    )
    parser.add_argument("input_file", help="Path to the input parquet file")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Optional output path. Defaults to INPUT_sorted_opt_xyz.parquet and "
            "must not already exist."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ascending",
        action="store_true",
        help="Sort from smallest to largest number_of_atoms.",
    )
    group.add_argument(
        "--descending",
        action="store_true",
        help="Sort from largest to smallest number_of_atoms. This is the default.",
    )
    return parser.parse_args(argv)


def infer_output_path(input_path: Path) -> Path:
    """Infer a non-conflicting output path next to the input file."""
    candidate = input_path.with_name(f"{input_path.stem}_sorted_opt_xyz.parquet")
    counter = 1

    while candidate.exists() or candidate.resolve() == input_path.resolve():
        candidate = input_path.with_name(
            f"{input_path.stem}_sorted_opt_xyz_{counter}.parquet"
        )
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


def validate_schema(schema: pa.Schema, input_path: Path) -> None:
    """Validate required columns and sorting type."""
    missing_columns = [
        column for column in OUTPUT_COLUMNS if column not in schema.names
    ]
    if missing_columns:
        formatted = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s) in {input_path}: {formatted}")

    for column in (SORT_COLUMN, TIE_BREAK_COLUMN):
        sort_type = schema.field(column).type
        if not (pa.types.is_integer(sort_type) or pa.types.is_floating(sort_type)):
            raise TypeError(f"{column} must be numeric, found {sort_type}")


def collect_distribution_stats(table: pa.Table) -> DistributionStats:
    """Collect row counts and per-value frequencies for number_of_atoms."""
    values = table[SORT_COLUMN].to_pylist()
    non_null_values = [value for value in values if value is not None]
    counts = sorted(Counter(non_null_values).items(), key=lambda item: item[0])
    total_rows = len(values)
    null_rows = total_rows - len(non_null_values)

    if not non_null_values:
        return DistributionStats(
            total_rows=total_rows,
            null_rows=null_rows,
            min_atoms=None,
            max_atoms=None,
            mean_atoms=None,
            counts=[],
        )

    return DistributionStats(
        total_rows=total_rows,
        null_rows=null_rows,
        min_atoms=min(non_null_values),
        max_atoms=max(non_null_values),
        mean_atoms=sum(non_null_values) / len(non_null_values),
        counts=counts,
    )


def format_atom_count(value: Any) -> str:
    """Format atom counts without trailing decimals for integer-like values."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def print_distribution_stats(stats: DistributionStats) -> None:
    """Print a compact atom-count distribution report."""
    non_null_rows = stats.total_rows - stats.null_rows

    print()
    print("number_of_atoms distribution:")
    print(f"  Rows       : {stats.total_rows:,}")
    print(f"  Non-null   : {non_null_rows:,}")
    print(f"  Null       : {stats.null_rows:,}")
    if non_null_rows:
        print(f"  Min        : {format_atom_count(stats.min_atoms)}")
        print(f"  Max        : {format_atom_count(stats.max_atoms)}")
        print(f"  Mean       : {stats.mean_atoms:.2f}")
        print()
        print(f"  {'Atoms':>10}  {'Rows':>10}  {'Percent':>8}")
        print(f"  {'-' * 10}  {'-' * 10}  {'-' * 8}")
        for atom_count, count in sorted(
            stats.counts,
            key=lambda item: item[0],
            reverse=True,
        ):
            pct = count / non_null_rows * 100
            print(f"  {format_atom_count(atom_count):>10}  {count:>10,}  {pct:>7.1f}%")


def write_sorted_opt_xyz_parquet(
    input_file: str,
    output_file: Optional[str] = None,
    descending: bool = True,
) -> SortResult:
    """Write a sorted parquet file containing selected structure columns."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    output_path = resolve_output_path(input_path, output_file)
    parquet_file = pq.ParquetFile(input_path)
    validate_schema(parquet_file.schema_arrow, input_path)
    table = parquet_file.read(columns=OUTPUT_COLUMNS)
    stats = collect_distribution_stats(table)

    order = "descending" if descending else "ascending"
    sorted_indices = pc.sort_indices(
        table,
        sort_keys=[(SORT_COLUMN, order), (TIE_BREAK_COLUMN, order)],
        null_placement="at_end",
    )
    sorted_table = table.take(sorted_indices)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        sorted_table,
        output_path,
        compression="zstd",
        use_dictionary=True,
    )
    return SortResult(output_path=output_path, stats=stats)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the parquet sort CLI."""
    args = parse_args(argv)
    result = write_sorted_opt_xyz_parquet(
        args.input_file,
        args.output,
        descending=not args.ascending,
    )
    print(f"Wrote sorted opt_xyz parquet: {result.output_path}")
    print_distribution_stats(result.stats)
    return 0


def run_cli() -> int:
    """CLI wrapper with consistent user-facing error handling."""
    try:
        return main()
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
