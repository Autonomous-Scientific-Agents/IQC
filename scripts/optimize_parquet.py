#!/usr/bin/env python3
"""
Script to optimize parquet files by moving constant columns to metadata.

If all rows have the same value for a column (or all are null), that column
is removed from the data and its value is stored in the file metadata instead.
"""
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def is_constant_column(column: pa.ChunkedArray) -> tuple[bool, Any]:
    """
    Check if a column has constant values.
    
    Returns:
        (is_constant, constant_value) tuple
        - is_constant: True if all non-null values are the same
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
        print(f"Error: File '{input_path}' does not exist.")
        sys.exit(1)
    
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
        return
    
    print(f"\nFound {len(constant_columns)} constant column(s)")
    print(f"Keeping {len(columns_to_keep)} variable column(s)")
    
    # Create new table without constant columns
    new_table = table.select(columns_to_keep)
    
    # Prepare metadata
    # Get existing metadata if any
    existing_metadata = table.schema.metadata or {}
    
    # Add constant columns to metadata
    constant_metadata = json.dumps(constant_columns)
    new_metadata = {
        **existing_metadata,
        b"constant_columns": constant_metadata.encode("utf-8")
    }
    
    # Create new schema with metadata
    new_schema = new_table.schema.with_metadata(new_metadata)
    new_table = new_table.cast(new_schema)
    
    # Write the optimized parquet file
    print(f"\nWriting optimized file...")
    pq.write_table(
        new_table,
        output_path,
        compression="zstd",
        use_dictionary=True,
    )
    
    # Compare file sizes
    original_size = input_path.stat().st_size
    new_size = output_path.stat().st_size
    savings = original_size - new_size
    savings_pct = (savings / original_size * 100) if original_size > 0 else 0
    
    print(f"\nFile size comparison:")
    print(f"  Original: {original_size:,} bytes ({original_size / (1024*1024):.2f} MB)")
    print(f"  Optimized: {new_size:,} bytes ({new_size / (1024*1024):.2f} MB)")
    print(f"  Savings: {savings:,} bytes ({savings_pct:.1f}%)")
    print(f"\nDone! Constant columns stored in metadata.")


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


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python optimize_parquet.py input.parquet [output.parquet]")
        print("\nIf output is not specified, the input file will be overwritten.")
        sys.exit(1)
    
    output_file = sys.argv[2] if len(sys.argv) == 3 else None
    optimize_parquet(sys.argv[1], output_file)

