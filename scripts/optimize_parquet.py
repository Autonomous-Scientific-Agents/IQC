#!/usr/bin/env python3
"""
Script to optimize parquet files by moving constant columns to metadata.

If all rows have the same value for a column (or all are null), that column
is removed from the data and its value is stored in the file metadata instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def is_constant_column(column: pa.ChunkedArray) -> tuple[bool, Any]:
    """
    Check if a column has constant values.

    Returns:
        (is_constant, constant_value) tuple
        - is_constant: True if all values are the same, including nulls
        - constant_value: The constant value, or None if all values are null
    """
    if len(column) == 0:
        return True, None

    # Convert ChunkedArray to regular Array and then to Python list
    # This handles nulls automatically (they become None)
    array = column.combine_chunks()
    values = array.to_pylist()

    # Filter out None values (nulls)
    non_null_values = [v for v in values if v is not None]

    # If all values are null, it's constant
    if len(non_null_values) == 0:
        return True, None

    # Nulls carry per-row information; moving [None, value] to metadata
    # would incorrectly fill the missing value in every downstream reader.
    if len(non_null_values) != len(values):
        return False, None

    # Get unique non-null values
    # Handle complex types by converting to JSON string for comparison
    unique_values = set()
    for val in non_null_values:
        # Handle complex types by converting to JSON string
        if isinstance(val, (dict, list)):
            val_key = json.dumps(val, sort_keys=True)
        else:
            val_key = val
        unique_values.add(val_key)
        # Early exit if we find more than one unique value
        if len(unique_values) > 1:
            return False, None

    # If we only have one unique value, return the first non-null value
    return True, non_null_values[0]


def optimize_parquet(input_file: str, output_file: str = None):
    """
    Read a parquet file and move constant columns to metadata.

    Args:
        input_file: Path to input parquet file
        output_file: Optional path to output file (default: overwrites input)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"File '{input_path}' does not exist.")

    if output_file is None:
        output_path = input_path
        print(f"Reading: {input_path}")
        print(f"Output: {output_path} (overwriting)")
    else:
        output_path = Path(output_file)
        print(f"Reading: {input_path}")
        print(f"Output: {output_path}")

    # Read the parquet file
    parquet_file = pq.ParquetFile(input_path)
    table = parquet_file.read()
    original_size = input_path.stat().st_size

    print(f"\nOriginal file: {len(table):,} rows, {len(table.column_names)} columns")

    # Find constant columns
    constant_columns = {}
    columns_to_keep = []

    for col_name in table.column_names:
        col = table[col_name]
        is_constant, constant_value = is_constant_column(col)

        if is_constant:
            constant_columns[col_name] = constant_value
            print(f"  Constant column: {col_name} = {constant_value}")
        else:
            columns_to_keep.append(col_name)

    if not constant_columns:
        print("\nNo constant columns found. File is already optimized.")
        if output_path == input_path:
            return

    if not columns_to_keep:
        # Every column is constant (routine for single-row chunk files).
        # Dropping them all would produce a 0-column table whose row count is
        # lost on the parquet round-trip — destroying the data when the
        # default in-place mode overwrites the input. Keep the first column
        # in the data to anchor the rows.
        anchor = table.column_names[0]
        constant_columns.pop(anchor)
        columns_to_keep = [anchor]
        print(
            f"\nAll columns are constant; keeping '{anchor}' in the data "
            "to preserve the rows."
        )
        if not constant_columns:
            print("Nothing left to move to metadata. File is already optimized.")
            if output_path == input_path:
                return

    print(f"\nFound {len(constant_columns)} constant column(s)")
    print(f"Keeping {len(columns_to_keep)} variable column(s)")

    # Create new table without constant columns
    new_table = table.select(columns_to_keep)

    # Prepare metadata
    # Get existing metadata if any
    existing_metadata = table.schema.metadata or {}

    # Merge with constants stored by an earlier optimize pass — replacing the
    # key wholesale would discard the previously moved column values.
    previous_constants = {}
    if b"constant_columns" in existing_metadata:
        try:
            previous_constants = json.loads(
                existing_metadata[b"constant_columns"].decode("utf-8")
            )
        except (ValueError, UnicodeDecodeError):
            previous_constants = {}

    # Add constant columns to metadata
    constant_metadata = json.dumps({**previous_constants, **constant_columns})
    new_metadata = {
        **existing_metadata,
        b"constant_columns": constant_metadata.encode("utf-8"),
    }

    # Create new schema with metadata
    new_schema = new_table.schema.with_metadata(new_metadata)
    new_table = new_table.cast(new_schema)

    print("\nWriting optimized file...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename so a failure mid-write (disk full,
    # Ctrl-C) cannot truncate the input when overwriting in place.
    fd, tmp_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(
            new_table,
            tmp_path,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Compare file sizes
    new_size = output_path.stat().st_size
    savings = original_size - new_size
    savings_pct = (savings / original_size * 100) if original_size > 0 else 0

    print("\nFile size comparison:")
    print(f"  Original: {original_size:,} bytes ({original_size / (1024*1024):.2f} MB)")
    print(f"  Optimized: {new_size:,} bytes ({new_size / (1024*1024):.2f} MB)")
    print(f"  Savings: {savings:,} bytes ({savings_pct:.1f}%)")
    print("\nDone! Constant columns stored in metadata.")
    return 0


def read_constant_columns(parquet_file: str) -> dict:
    """
    Read constant columns from a parquet file's metadata.

    Returns:
        Dictionary of constant column names to their values
    """
    parquet_path = Path(parquet_file)
    parquet_file_obj = pq.ParquetFile(parquet_path)
    table = parquet_file_obj.read()

    metadata = table.schema.metadata
    if metadata and b"constant_columns" in metadata:
        constant_str = metadata[b"constant_columns"].decode("utf-8")
        return json.loads(constant_str)

    return {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Move constant parquet columns into file metadata."
    )
    parser.add_argument("input_file", help="Path to input parquet file")
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Optional output path. If omitted, the input file is overwritten.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return optimize_parquet(args.input_file, args.output_file) or 0


def run_cli() -> int:
    try:
        return main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
