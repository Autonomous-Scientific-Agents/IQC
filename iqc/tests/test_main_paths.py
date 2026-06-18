import json

import pyarrow.parquet as pq

from iqc.cli import get_args
from iqc.databasetools import calculation_key_from_record
from iqc.main import (
    _default_skip_existing_sources,
    _make_run_id,
    _rank_output_parent,
    _unique_child_path,
    build_completed_calculation_index,
    convert_jsonl_results_to_parquet,
    validate_input_args,
)


def calculation_record(index, **overrides):
    record = {
        "initial_xyz": f"1\nmolecule {index}\nH {float(index):.8f} 0.0 0.0\n",
        "calculator": "mace",
        "model": "large",
        "task": "single",
        "params": {"cutoff": index % 3},
        "energy_eV": -float(index),
    }
    record.update(overrides)
    return record


def test_unique_child_path_returns_absolute_unused_path(tmp_path):
    path = _unique_child_path(tmp_path, "tmp_single_0_run")

    assert path == str(tmp_path / "tmp_single_0_run")


def test_unique_child_path_adds_suffix_for_existing_path(tmp_path):
    existing = tmp_path / "tmp_single_0_run"
    existing.mkdir()

    path = _unique_child_path(tmp_path, "tmp_single_0_run")

    assert path == str(tmp_path / "tmp_single_0_run_1")


def test_rank_output_parent_is_current_directory_tmp(tmp_path):
    assert _rank_output_parent(tmp_path) == tmp_path / "tmp"


def test_make_run_id_has_timestamp_and_random_suffix():
    run_id = _make_run_id()
    stamp, token = run_id.rsplit("_", 1)

    assert len(stamp) == len("YYYYMMDD_HHMMSS")
    assert stamp.replace("_", "").isdigit()
    assert len(token) == 8
    assert all(char in "0123456789abcdef" for char in token)


def test_default_skip_existing_sources_finds_iqc_result_files(tmp_path):
    combined = tmp_path / "iqc_single_results_20260101_000000_abcd1234.jsonl"
    combined.write_text("", encoding="utf-8")
    old_tmp_dir = tmp_path / "tmp_single_0_20260101_000000_abcd1234"
    old_tmp_dir.mkdir()
    old_per_rank = old_tmp_dir / "water_single_20260101_000001_0.json"
    old_per_rank.write_text("{}", encoding="utf-8")
    new_tmp_dir = tmp_path / "tmp" / "tmp_single_1_20260101_000000_abcd1234"
    new_tmp_dir.mkdir(parents=True)
    new_per_rank = new_tmp_dir / "methane_single_20260101_000001_1.json"
    new_per_rank.write_text("{}", encoding="utf-8")
    ignored = tmp_path / "other.json"
    ignored.write_text("{}", encoding="utf-8")

    assert _default_skip_existing_sources(tmp_path) == [
        combined,
        new_per_rank,
        old_per_rank,
    ]


def test_validate_input_args_rejects_jsonl_as_xyz():
    args = get_args(["--xyz", "iqc_single_results_20260101_000000.jsonl"])

    message = validate_input_args(args)

    assert "looks like a tabular/result file" in message
    assert "Use --input FILE" in message


def test_build_completed_calculation_index_reads_json_and_jsonl(tmp_path):
    record_from_json = calculation_record(1)
    record_from_jsonl = calculation_record(2)
    json_file = tmp_path / "one.json"
    jsonl_file = tmp_path / "many.jsonl"
    json_file.write_text(json.dumps(record_from_json), encoding="utf-8")
    jsonl_file.write_text(
        "\n".join(
            [
                json.dumps(record_from_jsonl),
                "not json",
                json.dumps({"initial_xyz": "missing required fields"}),
            ]
        ),
        encoding="utf-8",
    )

    index, summary = build_completed_calculation_index([json_file, jsonl_file])

    assert calculation_key_from_record(record_from_json) in index
    assert calculation_key_from_record(record_from_jsonl) in index
    assert summary == {
        "sources": 2,
        "files": 2,
        "records": 2,
        "invalid": 2,
        "ok": 2,
        "error": 0,
    }
    # Both records succeeded; the index now exposes status per F5.
    assert index[calculation_key_from_record(record_from_json)] == "ok"
    assert index[calculation_key_from_record(record_from_jsonl)] == "ok"


def test_convert_jsonl_results_to_parquet_writes_neighbor_file(tmp_path):
    jsonl_file = tmp_path / "iqc_single_results_20260101_000000_abcd1234.jsonl"
    jsonl_file.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "task": "single", "energy_eV": -1.0}),
                json.dumps({"id": 2, "task": "single", "energy_eV": -2.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parquet_file = convert_jsonl_results_to_parquet(jsonl_file)
    table = pq.read_table(parquet_file)

    expected_file = tmp_path / "iqc_single_results_20260101_000000_abcd1234.parquet"
    assert parquet_file == expected_file
    assert table.num_rows == 2
    assert table.column_names == ["id", "task", "energy_eV"]
