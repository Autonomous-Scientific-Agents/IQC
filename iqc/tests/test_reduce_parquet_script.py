"""Smoke tests for parquet utility scripts."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import (
    jsonl2parquet,
    optimize_parquet,
    read_parquet,
    reduce_parquet,
    sort_opt_xyz_parquet,
)


def write_sample_parquet(path: Path) -> pa.Table:
    """Create a small parquet file for exercising the script."""
    table = pa.table(
        {
            "id": [1, 2, 3],
            "payload": ["alpha", "beta", "gamma"],
            "notes": ["keep", None, "keep"],
        }
    )
    pq.write_table(table, path, compression="snappy")
    return table


def test_main_lists_columns_by_default(tmp_path, capsys):
    """Calling the script with only an input file should print column stats."""
    input_path = tmp_path / "sample.parquet"
    write_sample_parquet(input_path)

    exit_code = reduce_parquet.main([str(input_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Columns:" in captured.out
    assert "Idx" in captured.out
    assert "payload" in captured.out
    assert "notes" in captured.out
    assert "Use --drop IDX [IDX ...]" in captured.out


def test_main_drops_columns_and_writes_unique_output(tmp_path):
    """Dropping columns should create a new parquet file without overwriting."""
    input_path = tmp_path / "sample.parquet"
    table = write_sample_parquet(input_path)

    existing_output = tmp_path / "sample_reduced.parquet"
    pq.write_table(table.select(["id"]), existing_output, compression="snappy")

    exit_code = reduce_parquet.main([str(input_path), "--drop", "1,2"])

    output_path = tmp_path / "sample_reduced_1.parquet"
    reduced_table = pq.read_table(output_path)

    assert exit_code == 0
    assert input_path.exists()
    assert output_path.exists()
    assert reduced_table.column_names == ["id"]
    assert reduced_table.num_rows == table.num_rows


def test_jsonl2parquet_handles_missing_task_specific_fields(tmp_path):
    """JSONL conversion should preserve fields that only appear in some rows."""
    input_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "results.parquet"
    input_path.write_text(
        "\n".join(
            [
                '{"id": 1, "task": "thermo", "energy": -1.2, "params": {"drop": true}}',
                '{"id": 2, "task": "ir", "frequency": 123.4, "calculator_name": "uma"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = jsonl2parquet.main([str(input_path), str(output_path)])
    table = pq.read_table(output_path)

    assert exit_code == 0
    assert table.num_rows == 2
    assert set(table.column_names) == {"id", "task", "energy", "frequency"}
    assert "params" not in table.column_names
    assert "calculator_name" not in table.column_names
    assert table["energy"].to_pylist() == [-1.2, None]
    assert table["frequency"].to_pylist() == [None, 123.4]


def test_optimize_and_read_parquet_cli_entry_points(tmp_path, capsys):
    """Optimize should move constant columns to metadata readable by stats CLI."""
    input_path = tmp_path / "raw.parquet"
    output_path = tmp_path / "optimized.parquet"
    table = pa.table(
        {
            "id": [1, 2, 3],
            "calculator": ["uma", "uma", "uma"],
            "energy": [-1.0, -2.0, -3.0],
        }
    )
    pq.write_table(table, input_path, compression="snappy")

    optimize_exit = optimize_parquet.main([str(input_path), str(output_path)])
    optimized = pq.read_table(output_path)
    constants = optimize_parquet.read_constant_columns(str(output_path))
    read_exit = read_parquet.main([str(output_path)])

    captured = capsys.readouterr()
    assert optimize_exit == 0
    assert read_exit == 0
    assert optimized.column_names == ["id", "energy"]
    assert constants == {"calculator": "uma"}
    assert "Constant Columns (stored in metadata):" in captured.out


def test_sort_opt_xyz_parquet_writes_sorted_structure_columns(tmp_path, capsys):
    """The sorter should order by atom count and keep structure columns."""
    input_path = tmp_path / "results.parquet"
    output_path = tmp_path / "sorted.parquet"
    table = pa.table(
        {
            "xyz_file": [
                "six_lower_electrons.xyz",
                "smallest.xyz",
                "middle.xyz",
                "six_higher_electrons.xyz",
            ],
            "number_of_atoms": [6, 1, 3, 6],
            "number_of_electrons": [30, 2, 14, 34],
            "formula": ["C6", "H", "C3H2", "C6H4"],
            "unique_name": [
                "six_lower_electrons",
                "smallest",
                "middle",
                "six_higher_electrons",
            ],
            "opt_xyz": [
                "six lower electrons",
                "one atom",
                "three atoms",
                "six higher electrons",
            ],
            "extra": ["drop", "drop", "drop", "drop"],
        }
    )
    pq.write_table(table, input_path, compression="snappy")

    exit_code = sort_opt_xyz_parquet.main([str(input_path), "-o", str(output_path)])
    captured = capsys.readouterr()
    sorted_table = pq.read_table(output_path)

    assert exit_code == 0
    assert sorted_table.column_names == [
        "xyz_file",
        "number_of_atoms",
        "number_of_electrons",
        "formula",
        "unique_name",
        "opt_xyz",
    ]
    assert sorted_table["xyz_file"].to_pylist() == [
        "six_higher_electrons.xyz",
        "six_lower_electrons.xyz",
        "middle.xyz",
        "smallest.xyz",
    ]
    assert sorted_table["number_of_atoms"].to_pylist() == [6, 6, 3, 1]
    assert sorted_table["number_of_electrons"].to_pylist() == [34, 30, 14, 2]
    assert sorted_table["opt_xyz"].to_pylist() == [
        "six higher electrons",
        "six lower electrons",
        "three atoms",
        "one atom",
    ]
    assert "number_of_atoms distribution:" in captured.out
    assert "Rows       : 4" in captured.out
    assert "Max        : 6" in captured.out
    assert "Mean       : 4.00" in captured.out


def test_pyproject_exposes_script_entry_points():
    """The utility scripts should be available as package console commands."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert 'iqc-jsonl2parquet = "scripts.jsonl2parquet:run_cli"' in text
    assert 'iqc-optimize-parquet = "scripts.optimize_parquet:run_cli"' in text
    assert 'iqc-read-parquet = "scripts.read_parquet:run_cli"' in text
    assert 'iqc-reduce-parquet = "scripts.reduce_parquet:run_cli"' in text
    assert (
        'iqc-sort-opt-xyz-parquet = "scripts.sort_opt_xyz_parquet:run_cli"'
        in text
    )
