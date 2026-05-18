import json

from iqc.databasetools import calculation_key_from_record
from iqc.main import (
    _default_skip_existing_sources,
    _make_run_id,
    _unique_child_path,
    build_completed_calculation_index,
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
    tmp_dir = tmp_path / "tmp_single_0_20260101_000000_abcd1234"
    tmp_dir.mkdir()
    per_rank = tmp_dir / "water_single_20260101_000001_0.json"
    per_rank.write_text("{}", encoding="utf-8")
    ignored = tmp_path / "other.json"
    ignored.write_text("{}", encoding="utf-8")

    assert _default_skip_existing_sources(tmp_path) == [combined, per_rank]


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
    assert summary == {"sources": 2, "files": 2, "records": 2, "invalid": 2}
