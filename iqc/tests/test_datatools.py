import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from iqc.cli import get_args
from iqc.datatools import (
    inspect_data_file,
    read_smiles_column_records,
    read_xyz_column_records,
    run_input_inspection,
)
from iqc.main import get_structure_input_mode, validate_input_args


def test_cli_marks_input_only_for_short_and_long_options():
    short_args = get_args(["-i", "data.csv"])
    long_args = get_args(["--input", "data.csv"])
    mixed_args = get_args(["--input", "data.csv", "--task", "single"])
    xyz_calculation_args = get_args(["--input", "data.csv", "--xyz", "structure"])
    smiles_calculation_args = get_args(["--input", "data.csv", "--smiles", "smiles"])

    assert short_args.input == "data.csv"
    assert short_args.input_only is True
    assert short_args.input_xyz_column is False
    assert short_args.input_smiles_column is False
    assert long_args.input_only is True
    assert mixed_args.input_only is False
    assert mixed_args.input_xyz_column is False
    assert mixed_args.input_smiles_column is False
    assert xyz_calculation_args.input_only is False
    assert xyz_calculation_args.input_xyz_column is True
    assert xyz_calculation_args.input_smiles_column is False
    assert smiles_calculation_args.input_only is False
    assert smiles_calculation_args.input_xyz_column is False
    assert smiles_calculation_args.input_smiles_column is True


def test_data_input_modes_are_validated_without_mpi():
    inspect_args = get_args(["--input", "data.csv"])
    missing_column_args = get_args(["--input", "data.csv", "--task", "single"])
    both_columns_args = get_args(
        ["--input", "data.csv", "--xyz", "geometry", "--smiles", "smiles"]
    )
    xyz_args = get_args(["--input", "data.csv", "--xyz", "geometry"])
    smiles_args = get_args(["--input", "data.csv", "--smiles", "smiles"])
    direct_smiles_args = get_args(["--smiles", "O"])

    assert validate_input_args(inspect_args) is None
    assert "pass --xyz COLUMN or --smiles COLUMN" in validate_input_args(
        missing_column_args
    )
    assert "but not both" in validate_input_args(both_columns_args)
    assert validate_input_args(xyz_args) is None
    assert validate_input_args(smiles_args) is None
    assert get_structure_input_mode(xyz_args) == "data_xyz"
    assert get_structure_input_mode(smiles_args) == "data_smiles"
    assert get_structure_input_mode(direct_smiles_args) == "smiles"


def test_data_input_validation_sets_cli_exit_code(tmp_path):
    input_path = tmp_path / "molecules.csv"
    input_path.write_text("smiles\nO\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "iqc.main",
            "--input",
            str(input_path),
            "--task",
            "single",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "pass --xyz COLUMN or --smiles COLUMN" in result.stderr


def test_inspect_parquet_reports_schema_and_numeric_stats(tmp_path):
    input_path = tmp_path / "sample.parquet"
    table = pa.table(
        {
            "id": [1, 2, 3],
            "energy": [1.5, 2.5, None],
            "label": ["a", "b", "c"],
        }
    )
    pq.write_table(table, input_path)

    summary = inspect_data_file(input_path)

    assert summary.format_name == "parquet"
    assert summary.rows == 3
    assert summary.columns == 3
    assert [column.name for column in summary.column_summaries] == [
        "id",
        "energy",
        "label",
    ]
    energy = summary.column_summaries[1]
    assert energy.null_count == 1
    assert energy.min_value == "1.5"
    assert energy.max_value == "2.5"
    assert energy.mean_value == "2"


def test_run_input_inspection_prints_csv_summary(tmp_path, capsys):
    input_path = tmp_path / "sample.csv"
    input_path.write_text("name,count\nalpha,1\nbeta,\n", encoding="utf-8")

    exit_code = run_input_inspection(input_path)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Schema and Statistics:" in captured.out
    assert "Rows    : 2" in captured.out
    assert "name" in captured.out
    assert "count" in captured.out


def test_inspect_json_falls_back_when_fast_reader_cannot_parse(tmp_path):
    input_path = tmp_path / "sample.json"
    input_path.write_text('[{"name": "alpha", "count": 1}]', encoding="utf-8")

    summary = inspect_data_file(input_path)

    assert summary.format_name == "json"
    assert summary.engine == "pandas"
    assert summary.rows == 1
    assert [column.name for column in summary.column_summaries] == ["name", "count"]


def test_read_xyz_column_records_from_parquet(tmp_path):
    input_path = tmp_path / "molecules.parquet"
    water_xyz = "3\nwater\nO 0 0 0\nH 0 0 1\nH 1 0 0\n"
    hydrogen_xyz = "2\nhydrogen\nH 0 0 0\nH 0 0 0.7\n"
    table = pa.table(
        {
            "name": ["water", "hydrogen"],
            "geometry": [water_xyz, hydrogen_xyz],
        }
    )
    pq.write_table(table, input_path)

    records = read_xyz_column_records(input_path, "geometry")

    assert [record.row_index for record in records] == [0, 1]
    assert [record.xyz for record in records] == [water_xyz, hydrogen_xyz]


def test_read_xyz_column_records_from_csv_with_multiline_xyz(tmp_path):
    input_path = tmp_path / "molecules.csv"
    water_xyz = "3\nwater\nO 0 0 0\nH 0 0 1\nH 1 0 0\n"
    input_path.write_text(
        'name,geometry\nwater,"3\nwater\nO 0 0 0\nH 0 0 1\nH 1 0 0\n"\n',
        encoding="utf-8",
    )

    records = read_xyz_column_records(input_path, "geometry")

    assert len(records) == 1
    assert records[0].row_index == 0
    assert records[0].xyz == water_xyz


def test_read_smiles_column_records_from_parquet(tmp_path):
    input_path = tmp_path / "molecules.parquet"
    table = pa.table(
        {
            "name": ["water", "ethanol"],
            "smiles": [" O ", "CCO"],
        }
    )
    pq.write_table(table, input_path)

    records = read_smiles_column_records(input_path, "smiles")

    assert [record.row_index for record in records] == [0, 1]
    assert [record.smiles for record in records] == ["O", "CCO"]


def test_read_smiles_column_records_rejects_blank_values(tmp_path):
    input_path = tmp_path / "molecules.parquet"
    table = pa.table({"smiles": ["O", "   "]})
    pq.write_table(table, input_path)

    try:
        read_smiles_column_records(input_path, "smiles")
    except ValueError as exc:
        assert "blank SMILES value at row 1" in str(exc)
    else:
        raise AssertionError("Expected blank SMILES values to be rejected.")


def test_read_structure_column_records_report_missing_column(tmp_path):
    input_path = tmp_path / "molecules.parquet"
    table = pa.table({"name": ["water"], "smiles": ["O"]})
    pq.write_table(table, input_path)

    try:
        read_xyz_column_records(input_path, "geometry")
    except ValueError as exc:
        message = str(exc)
        assert "Column 'geometry' was not found" in message
        assert "name, smiles" in message
    else:
        raise AssertionError("Expected missing structure columns to be rejected.")


def test_read_xyz_column_records_rejects_empty_values(tmp_path):
    input_path = tmp_path / "molecules.parquet"
    table = pa.table({"geometry": ["3\nwater\nO 0 0 0\nH 0 0 1\nH 1 0 0\n", None]})
    pq.write_table(table, input_path)

    try:
        read_xyz_column_records(input_path, "geometry")
    except ValueError as exc:
        assert "empty XYZ value at row 1" in str(exc)
    else:
        raise AssertionError("Expected empty XYZ values to be rejected.")
