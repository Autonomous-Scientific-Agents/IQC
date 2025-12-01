#!/usr/bin/env python3
import json
import gzip
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Fields to drop from each JSON object
DROP_FIELDS = {
    "params",
    "optimization_params",
    "vibration_params",
    "thermo_params",
    "calculator_name",
    # "warnings",
    # "error",
    "trajectory_file",
    "optimized_geometry_file",
    #  "jmol_vib_modes_xyz",
    "vib_modes",
}

ROWS_PER_BATCH = 25_000  # adjust if you want larger/smaller memory usage


def infer_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".jsonl.gz"):
        out_name = name[:-8] + ".parquet"  # remove ".jsonl.gz"
    elif name.endswith(".jsonl"):
        out_name = name[:-6] + ".parquet"  # remove ".jsonl"
    else:
        # fallback: just append .parquet
        out_name = name + ".parquet"
    return input_path.with_name(out_name)


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    else:
        return open(path, "rt", encoding="utf-8")


def convert_jsonl_to_parquet(input_file: str, output_file: str = None):
    input_path = Path(input_file)
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = infer_output_path(input_path)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    rows = []
    writer = None
    total_rows = 0

    with open_maybe_gzip(input_path) as fin:
        for line_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  Skipping malformed JSON at line {line_idx}")
                continue

            # Drop unwanted fields
            for f in DROP_FIELDS:
                data.pop(f, None)

            rows.append(data)
            total_rows += 1

            if len(rows) >= ROWS_PER_BATCH:
                writer = write_batch(rows, writer, output_path)
                rows = []

            if total_rows % 100_000 == 0:
                print(f"Processed {total_rows:,} records...")

    # Last partial batch
    if rows:
        writer = write_batch(rows, writer, output_path)

    if writer is not None:
        writer.close()

    print(f"Done. Total rows written: {total_rows:,}")


def write_batch(rows, writer, output_path: Path):
    table = pa.Table.from_pylist(rows)

    if writer is None:
        writer = pq.ParquetWriter(
            output_path,
            schema=table.schema,
            compression="zstd",
            use_dictionary=True,
        )

    writer.write_table(table)
    return writer


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python jsonl_to_parquet.py input.jsonl [output.parquet]")
        sys.exit(1)

    output_file = sys.argv[2] if len(sys.argv) == 3 else None
    convert_jsonl_to_parquet(sys.argv[1], output_file)
