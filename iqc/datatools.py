"""Utilities for inspecting tabular data files."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence


class ColumnNotFoundError(ValueError):
    """Raised when a requested data column is absent."""


@dataclass(frozen=True)
class ColumnSummary:
    """Compact summary for one data column."""

    index: int
    name: str
    data_type: str
    null_count: int
    null_pct: float
    min_value: str = ""
    max_value: str = ""
    mean_value: str = ""


@dataclass(frozen=True)
class DataSummary:
    """File-level and column-level data summary."""

    path: Path
    format_name: str
    rows: int
    columns: int
    size_bytes: int
    engine: str
    column_summaries: Sequence[ColumnSummary]


@dataclass(frozen=True)
class XYZColumnRecord:
    """XYZ text and source row metadata from a data input file."""

    row_index: int
    xyz: str


@dataclass(frozen=True)
class SMILESColumnRecord:
    """SMILES text and source row metadata from a data input file."""

    row_index: int
    smiles: str


def format_bytes(size: int) -> str:
    """Format a byte count using compact binary units."""

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"  # unreachable; satisfies type checkers


def truncate(text: Any, width: int) -> str:
    """Return a display-safe, truncated string."""

    value = "" if text is None else str(text)
    if len(value) <= width:
        return value
    return value[: width - 3] + "..."


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return truncate(value, 24)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading this file requires pandas. Install pandas or use parquet/csv "
            "with pyarrow available."
        ) from exc
    return pd


def _read_arrow_table(path: Path, suffix: str):
    import pyarrow.csv as pacsv
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    import pyarrow.json as pajson
    import pyarrow.parquet as pq

    if suffix in {".parquet", ".pq"}:
        return pq.read_table(path), "parquet", "pyarrow"
    if suffix in {".csv", ".tsv", ".tab", ".txt"}:
        delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
        parse_options = pacsv.ParseOptions(
            delimiter=delimiter,
            newlines_in_values=True,
        )
        return (
            pacsv.read_csv(path, parse_options=parse_options),
            "delimited text",
            "pyarrow",
        )
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pajson.read_json(path), "json lines", "pyarrow"
    if suffix in {".feather", ".ftr"}:
        return feather.read_table(path), "feather", "pyarrow"
    if suffix in {".arrow", ".ipc"}:
        with ipc.open_file(path) as reader:
            return reader.read_all(), "arrow ipc", "pyarrow"
    raise ValueError(f"Unsupported input file type: {suffix or '<none>'}")


def _read_pandas_frame(path: Path, suffix: str):
    pd = _require_pandas()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path), "parquet", "pandas"
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path), "delimited text", "pandas"
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t"), "delimited text", "pandas"
    if suffix in {".xls", ".xlsx", ".xlsm", ".ods"}:
        return pd.read_excel(path), "excel", "pandas"
    if suffix == ".json":
        return pd.read_json(path), "json", "pandas"
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True), "json lines", "pandas"
    if suffix in {".feather", ".ftr"}:
        return pd.read_feather(path), "feather", "pandas"
    raise ValueError(f"Unsupported input file type: {suffix or '<none>'}")


def _ensure_column(
    column_name: str, available_columns: Sequence[str], path: Path
) -> None:
    if column_name in available_columns:
        return

    preview = ", ".join(available_columns[:12])
    if len(available_columns) > 12:
        preview += ", ..."
    raise ColumnNotFoundError(
        f"Column '{column_name}' was not found in {path}. "
        f"Available columns: {preview}"
    )


def _read_arrow_column(path: Path, suffix: str, column_name: str):
    import pyarrow.csv as pacsv
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    import pyarrow.json as pajson
    import pyarrow.parquet as pq

    if suffix in {".parquet", ".pq"}:
        parquet_file = pq.ParquetFile(path)
        _ensure_column(column_name, parquet_file.schema_arrow.names, path)
        table = pq.read_table(path, columns=[column_name])
        return table[column_name].to_pylist(), "pyarrow"

    if suffix in {".csv", ".tsv", ".tab", ".txt"}:
        delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
        parse_options = pacsv.ParseOptions(
            delimiter=delimiter,
            newlines_in_values=True,
        )
        convert_options = pacsv.ConvertOptions(include_columns=[column_name])
        table = pacsv.read_csv(
            path,
            parse_options=parse_options,
            convert_options=convert_options,
        )
        return table[column_name].to_pylist(), "pyarrow"

    if suffix in {".json", ".jsonl", ".ndjson"}:
        table = pajson.read_json(path)
        _ensure_column(column_name, table.column_names, path)
        return table[column_name].to_pylist(), "pyarrow"

    if suffix in {".feather", ".ftr"}:
        table = feather.read_table(path, columns=[column_name])
        return table[column_name].to_pylist(), "pyarrow"

    if suffix in {".arrow", ".ipc"}:
        with ipc.open_file(path) as reader:
            _ensure_column(column_name, reader.schema.names, path)
            table = reader.read_all()
        return table[column_name].to_pylist(), "pyarrow"

    raise ValueError(f"Unsupported input file type: {suffix or '<none>'}")


def _read_pandas_column(path: Path, suffix: str, column_name: str):
    pd = _require_pandas()

    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=[column_name])
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path, usecols=[column_name])
    elif suffix in {".tsv", ".tab"}:
        frame = pd.read_csv(path, sep="\t", usecols=[column_name])
    elif suffix in {".xls", ".xlsx", ".xlsm", ".ods"}:
        frame = pd.read_excel(path, usecols=[column_name])
    elif suffix == ".json":
        frame = pd.read_json(path)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    elif suffix in {".feather", ".ftr"}:
        frame = pd.read_feather(path, columns=[column_name])
    else:
        raise ValueError(f"Unsupported input file type: {suffix or '<none>'}")

    available_columns = [str(column) for column in frame.columns]
    _ensure_column(column_name, available_columns, path)
    return frame[column_name].tolist(), "pandas"


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (ImportError, TypeError, ValueError):
        pass
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _coerce_text_value(
    value: Any, row_index: int, column_name: str, value_label: str
) -> str:
    if _is_missing_value(value):
        raise ValueError(
            f"Column '{column_name}' contains an empty {value_label} value "
            f"at row {row_index}."
        )
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif not isinstance(value, str):
        value = str(value)

    if not value.strip():
        raise ValueError(
            f"Column '{column_name}' contains a blank {value_label} value "
            f"at row {row_index}."
        )
    # XYZ records preserve their original formatting (trailing newlines,
    # internal whitespace); SMILES are single-line and get trimmed.
    return value.strip() if value_label == "SMILES" else value


def _sort_key(value: Any):
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (1, float(value))
    if isinstance(value, (datetime, date)):
        return (2, value.isoformat())
    if isinstance(value, bytes):
        return (3, value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return (3, value)
    return (4, str(value))


def _sort_records(records, sort_values, sort_column: str, sort_order: str):
    if sort_column is None:
        return records
    if sort_order not in {"up", "down"}:
        raise ValueError("--sort_order must be 'up' or 'down'.")
    if len(records) != len(sort_values):
        raise ValueError(
            f"Sort column '{sort_column}' row count does not match structure rows."
        )

    indexed_records = []
    for record, sort_value in zip(records, sort_values):
        if _is_missing_value(sort_value):
            raise ValueError(
                f"Sort column '{sort_column}' contains an empty value at row "
                f"{record.row_index}."
            )
        indexed_records.append((_sort_key(sort_value), record))

    return [
        record
        for _key, record in sorted(
            indexed_records,
            key=lambda item: item[0],
            reverse=sort_order == "down",
        )
    ]


def _read_column_values(input_file: str | Path, column_name: str) -> list[Any]:
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    suffix = path.suffix.lower()
    try:
        values, _engine = _read_arrow_column(path, suffix, column_name)
    except ImportError:
        values, _engine = _read_pandas_column(path, suffix, column_name)
    except ColumnNotFoundError:
        raise
    except Exception:
        values, _engine = _read_pandas_column(path, suffix, column_name)

    if not values:
        raise ValueError(f"Column '{column_name}' in {path} does not contain any rows.")
    return values


def read_xyz_column_records(
    input_file: str | Path,
    column_name: str,
    sort_column: str | None = None,
    sort_order: str = "up",
) -> list[XYZColumnRecord]:
    """Read XYZ strings from a named column in a supported tabular data file."""

    values = _read_column_values(input_file, column_name)
    records = [
        XYZColumnRecord(
            row_index=row_index,
            xyz=_coerce_text_value(value, row_index, column_name, "XYZ"),
        )
        for row_index, value in enumerate(values)
    ]
    sort_values = (
        _read_column_values(input_file, sort_column)
        if sort_column is not None
        else None
    )
    return _sort_records(records, sort_values, sort_column, sort_order)


def read_smiles_column_records(
    input_file: str | Path,
    column_name: str,
    sort_column: str | None = None,
    sort_order: str = "up",
) -> list[SMILESColumnRecord]:
    """Read SMILES strings from a named column in a supported tabular data file."""

    values = _read_column_values(input_file, column_name)
    records = [
        SMILESColumnRecord(
            row_index=row_index,
            smiles=_coerce_text_value(value, row_index, column_name, "SMILES"),
        )
        for row_index, value in enumerate(values)
    ]
    sort_values = (
        _read_column_values(input_file, sort_column)
        if sort_column is not None
        else None
    )
    return _sort_records(records, sort_values, sort_column, sort_order)


def _arrow_column_summaries(table) -> list[ColumnSummary]:
    import pyarrow.compute as pc
    import pyarrow.types as patypes

    summaries = []
    rows = table.num_rows
    for index, field in enumerate(table.schema):
        column = table[field.name]
        null_count = column.null_count
        null_pct = (null_count / rows * 100) if rows else 0.0
        min_value = ""
        max_value = ""
        mean_value = ""

        if rows and (
            patypes.is_integer(field.type)
            or patypes.is_floating(field.type)
            or patypes.is_decimal(field.type)
            or patypes.is_temporal(field.type)
        ):
            try:
                min_max = pc.min_max(column).as_py()
                min_value = _format_value(min_max.get("min"))
                max_value = _format_value(min_max.get("max"))
            except Exception:
                pass
        if rows and (patypes.is_integer(field.type) or patypes.is_floating(field.type)):
            try:
                mean_value = _format_value(pc.mean(column).as_py())
            except Exception:
                pass

        summaries.append(
            ColumnSummary(
                index=index,
                name=field.name,
                data_type=str(field.type),
                null_count=null_count,
                null_pct=null_pct,
                min_value=min_value,
                max_value=max_value,
                mean_value=mean_value,
            )
        )
    return summaries


def _pandas_column_summaries(frame) -> list[ColumnSummary]:
    pd = _require_pandas()

    summaries = []
    rows = len(frame)
    for index, name in enumerate(frame.columns):
        series = frame[name]
        null_count = int(series.isna().sum())
        null_pct = (null_count / rows * 100) if rows else 0.0
        min_value = ""
        max_value = ""
        mean_value = ""

        if rows and (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
        ):
            non_null = series.dropna()
            if not non_null.empty:
                min_value = _format_value(non_null.min())
                max_value = _format_value(non_null.max())
                if pd.api.types.is_numeric_dtype(series):
                    mean_value = _format_value(non_null.mean())

        summaries.append(
            ColumnSummary(
                index=index,
                name=str(name),
                data_type=str(series.dtype),
                null_count=null_count,
                null_pct=null_pct,
                min_value=min_value,
                max_value=max_value,
                mean_value=mean_value,
            )
        )
    return summaries


def inspect_data_file(input_file: str | Path) -> DataSummary:
    """Read a supported data file and return schema plus basic statistics."""

    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    suffix = path.suffix.lower()
    try:
        table, format_name, engine = _read_arrow_table(path, suffix)
        rows = table.num_rows
        columns = table.num_columns
        column_summaries = _arrow_column_summaries(table)
    except Exception:
        frame, format_name, engine = _read_pandas_frame(path, suffix)
        rows = len(frame)
        columns = len(frame.columns)
        column_summaries = _pandas_column_summaries(frame)

    return DataSummary(
        path=path,
        format_name=format_name,
        rows=rows,
        columns=columns,
        size_bytes=path.stat().st_size,
        engine=engine,
        column_summaries=column_summaries,
    )


def print_data_summary(summary: DataSummary) -> None:
    """Print a human-readable schema and basic statistics report."""

    print(f"File    : {summary.path}")
    print(f"Format  : {summary.format_name}")
    print(f"Engine  : {summary.engine}")
    print(f"Rows    : {summary.rows:,}")
    print(f"Columns : {summary.columns:,}")
    print(f"Size    : {format_bytes(summary.size_bytes)}")
    print()
    print("Schema and Statistics:")
    header = (
        f"{'Idx':>3}  {'Name':<28}  {'Type':<20}  {'Nulls':>10}  "
        f"{'Null %':>7}  {'Min':>12}  {'Max':>12}  {'Mean':>12}"
    )
    print(header)
    print("-" * len(header))
    for column in summary.column_summaries:
        print(
            f"{column.index:>3}  "
            f"{truncate(column.name, 28):<28}  "
            f"{truncate(column.data_type, 20):<20}  "
            f"{column.null_count:>10,}  "
            f"{column.null_pct:>6.1f}%  "
            f"{truncate(column.min_value, 12):>12}  "
            f"{truncate(column.max_value, 12):>12}  "
            f"{truncate(column.mean_value, 12):>12}"
        )


def run_input_inspection(input_file: str | Path) -> int:
    """Inspect a data file and print user-facing errors consistently."""

    try:
        print_data_summary(inspect_data_file(input_file))
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
