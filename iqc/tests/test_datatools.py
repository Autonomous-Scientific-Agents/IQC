import pyarrow as pa
import pyarrow.parquet as pq

from iqc.cli import get_args
from iqc.datatools import inspect_data_file, run_input_inspection


def test_cli_marks_input_only_for_short_and_long_options():
    short_args = get_args(["-i", "data.csv"])
    long_args = get_args(["--input", "data.csv"])
    mixed_args = get_args(["--input", "data.csv", "--task", "single"])

    assert short_args.input == "data.csv"
    assert short_args.input_only is True
    assert long_args.input_only is True
    assert mixed_args.input_only is False


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
