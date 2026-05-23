#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def read_parquet_stats(parquet_file: str):
    """Read a parquet file and display basic statistics."""
    parquet_path = Path(parquet_file)

    if not parquet_path.exists():
        raise FileNotFoundError(f"File '{parquet_path}' does not exist.")

    print(f"Reading: {parquet_path}")
    print("-" * 60)

    # Read parquet file metadata
    parquet_file_obj = pq.ParquetFile(parquet_path)
    table = parquet_file_obj.read()

    # Basic file info
    file_size = parquet_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    print(f"File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
    print(f"Number of rows: {len(table):,}")
    print(f"Number of columns: {len(table.column_names)}")
    print()

    # Schema information
    print("Schema:")
    print("-" * 60)
    schema = table.schema
    for field in schema:
        field_name = field.name
        field_type = str(field.type)
        nullable = "nullable" if field.nullable else "not null"
        print(f"  {field_name:30s} {field_type:20s} ({nullable})")

    print()

    # Column statistics
    print("Column Statistics:")
    print("-" * 60)
    for col_name in table.column_names:
        col = table[col_name]
        null_count = col.null_count
        total_count = len(col)
        null_pct = (null_count / total_count * 100) if total_count > 0 else 0

        print(f"  {col_name:30s} nulls: {null_count:,} ({null_pct:.1f}%)")

    # Metadata
    metadata = parquet_file_obj.metadata
    print()
    print("File Metadata:")
    print("-" * 60)
    print(f"  Number of row groups: {metadata.num_row_groups}")
    print(f"  Created by: {metadata.created_by if metadata.created_by else 'unknown'}")
    print(
        f"  Compression: {metadata.row_group(0).column(0).compression if metadata.num_row_groups > 0 else 'unknown'}"
    )

    # Check for constant columns in schema metadata
    schema_metadata = table.schema.metadata
    if schema_metadata and b"constant_columns" in schema_metadata:
        constant_columns = json.loads(
            schema_metadata[b"constant_columns"].decode("utf-8")
        )
        print()
        print("Constant Columns (stored in metadata):")
        print("-" * 60)
        for col_name, col_value in constant_columns.items():
            # Truncate long values for display
            value_str = str(col_value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            print(f"  {col_name:30s} = {value_str}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Print parquet file statistics.")
    parser.add_argument("input_file", help="Path to input parquet file")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return read_parquet_stats(args.input_file)


def run_cli() -> int:
    try:
        return main()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
