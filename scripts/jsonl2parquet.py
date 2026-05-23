#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import gzip
import sys
from collections import OrderedDict
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


def iter_jsonl_records(input_path: Path, warn: bool = True):
    with open_maybe_gzip(input_path) as fin:
        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                if warn:
                    print(f"Warning: skipping malformed JSON at line {line_idx}")
                continue

            for field in DROP_FIELDS:
                data.pop(field, None)
            yield data


def collect_field_names_and_schema(input_path: Path) -> tuple[list[str], pa.Schema]:
    fields = OrderedDict()
    example = {}
    for data in iter_jsonl_records(input_path, warn=False):
        for key, value in data.items():
            fields.setdefault(key, None)
            if value is not None and key not in example:
                example[key] = value
    field_names = list(fields)
    schema_row = {field: example.get(field) for field in field_names}
    schema = pa.Table.from_pylist([schema_row]).schema
    return field_names, schema


def normalize_rows(rows: list[dict], field_names: list[str]) -> list[dict]:
    return [{field: row.get(field) for field in field_names} for row in rows]


def convert_jsonl_to_parquet(input_file: str, output_file: str = None) -> int:
    input_path = Path(input_file)
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = infer_output_path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    field_names, schema = collect_field_names_and_schema(input_path)
    if not field_names:
        raise ValueError(f"No JSON records found in {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    writer = None
    total_rows = 0

    for data in iter_jsonl_records(input_path):
        rows.append(data)
        total_rows += 1

        if len(rows) >= ROWS_PER_BATCH:
            writer = write_batch(rows, writer, output_path, field_names, schema)
            rows = []

        if total_rows % 100_000 == 0:
            print(f"Processed {total_rows:,} records...")

    # Last partial batch
    if rows:
        writer = write_batch(rows, writer, output_path, field_names, schema)

    if writer is not None:
        writer.close()

    print(f"Done. Total rows written: {total_rows:,}")
    return 0


def write_batch(rows, writer, output_path: Path, field_names: list[str], schema=None):
    table = pa.Table.from_pylist(normalize_rows(rows, field_names), schema=schema)

    if writer is None:
        writer = pq.ParquetWriter(
            output_path,
            schema=table.schema,
            compression="zstd",
            use_dictionary=True,
        )

    writer.write_table(table)
    return writer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert IQC JSONL or JSONL.GZ results to Parquet."
    )
    parser.add_argument("input_file", help="Path to input .jsonl or .jsonl.gz file")
    parser.add_argument("output_file", nargs="?", help="Optional output parquet path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return convert_jsonl_to_parquet(args.input_file, args.output_file)


def run_cli() -> int:
    try:
        return main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
