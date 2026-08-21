#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import gzip
import os
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
    # Infer a schema per batch and unify permissively across the whole file.
    # A single-example-per-field schema silently corrupted data: a field whose
    # first non-null value was a JSON integer got an int64 schema, truncating
    # every later float (-22.943... stored as -22), and a struct field typed
    # from its first occurrence dropped keys present only in later rows.
    # Permissive unification promotes int64+double -> double, unions struct
    # keys, and resolves null-typed fields (e.g. warnings=[] in the first
    # batch) against later batches that carry values.
    fields = OrderedDict()
    schemas = []
    batch = []

    def batch_schema(rows):
        # from_pylist infers the schema from the first row's keys only, so
        # normalize each row to the union of keys seen in this batch.
        batch_fields = OrderedDict()
        for row in rows:
            for key in row:
                batch_fields.setdefault(key, None)
        return pa.Table.from_pylist(
            normalize_rows(rows, list(batch_fields))
        ).schema

    for data in iter_jsonl_records(input_path, warn=False):
        for key in data:
            fields.setdefault(key, None)
        batch.append(data)
        if len(batch) >= ROWS_PER_BATCH:
            schemas.append(batch_schema(batch))
            batch = []
    if batch:
        schemas.append(batch_schema(batch))
    field_names = list(fields)
    if not schemas:
        return field_names, None
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    # Defensive fallback: if every observation of a list-typed field was
    # empty (so it still infers as list<null>), coerce to list<string>.
    # Real IQC list-typed fields ("warnings", etc.) are list<str>.
    schema = _coerce_null_list_to_string_list(schema)
    return field_names, schema


def _coerce_null_list_to_string_list(schema: pa.Schema) -> pa.Schema:
    """Replace any list<null> with list<string> (recursing through nested lists)."""

    def fix(t: pa.DataType) -> pa.DataType:
        if pa.types.is_list(t) or pa.types.is_large_list(t):
            value = t.value_type
            if pa.types.is_null(value):
                return pa.list_(pa.string())
            return pa.list_(fix(value))
        return t

    return pa.schema([pa.field(f.name, fix(f.type), nullable=True) for f in schema])


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

    # Stream into a temp file and rename on success so a failure mid-write
    # (disk full, kill) never leaves a truncated parquet at the target path.
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    rows = []
    writer = None
    total_rows = 0

    try:
        for data in iter_jsonl_records(input_path):
            rows.append(data)
            total_rows += 1

            if len(rows) >= ROWS_PER_BATCH:
                writer = write_batch(rows, writer, tmp_path, field_names, schema)
                rows = []

            if total_rows % 100_000 == 0:
                print(f"Processed {total_rows:,} records...")

        # Last partial batch
        if rows:
            writer = write_batch(rows, writer, tmp_path, field_names, schema)

        if writer is not None:
            writer.close()
            writer = None
            os.replace(tmp_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()

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
