"""Tests for iqc.main behaviors that span result-record indexing and CLI."""

import json

from iqc.cli import get_args
from iqc.databasetools import calculation_key_from_record
from iqc.main import build_completed_calculation_index


def _record(index, **overrides):
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


def test_build_index_marks_status_per_record(tmp_path):
    ok_record = _record(1)
    err_record = _record(2, single_error="SCF failed to converge")
    jsonl_file = tmp_path / "results.jsonl"
    jsonl_file.write_text(
        "\n".join([json.dumps(ok_record), json.dumps(err_record)]),
        encoding="utf-8",
    )

    index, summary = build_completed_calculation_index([jsonl_file])

    ok_key = calculation_key_from_record(ok_record)
    err_key = calculation_key_from_record(err_record)
    assert index[ok_key] == "ok"
    assert index[err_key] == "error"
    assert summary["ok"] == 1
    assert summary["error"] == 1
    assert summary["records"] == 2


def test_build_index_include_errors_false_excludes_failures(tmp_path):
    ok_record = _record(3)
    err_record = _record(4, single_error="boom")
    jsonl_file = tmp_path / "results.jsonl"
    jsonl_file.write_text(
        "\n".join([json.dumps(ok_record), json.dumps(err_record)]),
        encoding="utf-8",
    )

    index, summary = build_completed_calculation_index(
        [jsonl_file], include_errors=False
    )

    assert calculation_key_from_record(ok_record) in index
    assert calculation_key_from_record(err_record) not in index
    # Summary still records what was scanned; it just isn't added to the index.
    assert summary["ok"] == 1
    assert summary["error"] == 1


def test_build_index_parsl_retries_exhausted_is_error(tmp_path):
    failed = _record(
        5,
        single_error="row failed after retries",
        parsl_retries_exhausted=True,
    )
    jsonl_file = tmp_path / "results.jsonl"
    jsonl_file.write_text(json.dumps(failed), encoding="utf-8")

    index, _ = build_completed_calculation_index([jsonl_file])
    assert index[calculation_key_from_record(failed)] == "error"

    index_ok_only, _ = build_completed_calculation_index(
        [jsonl_file], include_errors=False
    )
    assert calculation_key_from_record(failed) not in index_ok_only


def test_build_index_prefers_ok_over_error_for_same_key(tmp_path):
    # First we saw a failure; a later retry succeeded — the success should win
    # so a future run doesn't waste cycles recomputing it.
    err_record = _record(6, single_error="transient")
    ok_record = _record(6)  # same key
    jsonl_file = tmp_path / "results.jsonl"
    jsonl_file.write_text(
        "\n".join([json.dumps(err_record), json.dumps(ok_record)]),
        encoding="utf-8",
    )

    index, _ = build_completed_calculation_index([jsonl_file])
    assert index[calculation_key_from_record(ok_record)] == "ok"


def test_cli_retry_failed_only_defaults_off_and_parses_true():
    args_default = get_args(["--xyz", "water.xyz"])
    assert getattr(args_default, "retry_failed_only", False) is False

    args_retry = get_args(["--xyz", "water.xyz", "--retry-failed-only"])
    assert args_retry.retry_failed_only is True
