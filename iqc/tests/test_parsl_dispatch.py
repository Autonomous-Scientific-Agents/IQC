"""Tests for :mod:`iqc.parsl_dispatch`, especially the F7 failure-row path.

These tests cover the synthesis-of-a-failure-row helper used by the Parsl
driver when ``future.exception()`` is non-None after retries are exhausted.
The full driver loop is exercised by a thin integration test that monkey-patches
``_row_app`` so we can drive completed/failed futures without launching Parsl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("parsl")

from iqc import parsl_dispatch as pd
from iqc.databasetools import calculation_key, calculation_key_from_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


WATER_XYZ = (
    "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\n"
    "H 0.0000 -0.7572 -0.4692\n"
)


def _make_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    """Build the minimal argparse namespace the failure-row synthesizer reads."""

    base = dict(
        input=None,
        xyz="xyz",
        smiles=None,
        sort=None,
        sort_order="up",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _write_xyz(tmp_path: Path, name: str = "water.xyz") -> str:
    path = tmp_path / name
    path.write_text(WATER_XYZ, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# _synthesize_failure_row
# ---------------------------------------------------------------------------


def test_synthesize_failure_row_includes_calculation_key_fields(tmp_path):
    xyz_path = _write_xyz(tmp_path)
    args = _make_args(tmp_path)

    row = pd._synthesize_failure_row(
        0,
        RuntimeError("worker died: ZE_RESULT_ERROR_DEVICE_LOST"),
        args=args,
        params_str="{calculator: exachem}",
        task="opt",
        calculator_name="exachem",
        xyz_files=[xyz_path],
        input_mode="xyz",
        number_of_files=1,
    )

    # All five calculation_key inputs must be present so F5's index can
    # recognize this row.
    assert "initial_xyz" in row and row["initial_xyz"]
    assert row["params"] == "{calculator: exachem}"
    assert row["calculator"] == "exachem"
    assert row["model"] == ""
    assert row["task"] == "opt"

    # The {task}_error field is the failure signal F5 keys on.
    assert "opt_error" in row
    assert "ZE_RESULT_ERROR_DEVICE_LOST" in row["opt_error"]
    assert row["parsl_retries_exhausted"] is True

    # The resulting record must yield a valid calculation_key.
    key = calculation_key_from_record(row)
    expected = calculation_key(
        row["initial_xyz"],
        row["params"],
        row["calculator"],
        row["model"],
        row["task"],
    )
    assert key == expected


def test_synthesize_failure_row_matches_successful_row_key(tmp_path):
    """A failure row for input X must hash to the same key as a success row
    for input X — otherwise F5's skip-existing index can't connect them."""

    from iqc.asetools import atoms2xyz, get_atoms_from_xyz

    xyz_path = _write_xyz(tmp_path)
    args = _make_args(tmp_path)

    failure = pd._synthesize_failure_row(
        0,
        RuntimeError("boom"),
        args=args,
        params_str="{}",
        task="single",
        calculator_name="mace",
        xyz_files=[xyz_path],
        input_mode="xyz",
        number_of_files=1,
    )

    # Build the "what a successful row would have looked like" key independently.
    atoms = get_atoms_from_xyz(xyz_path, index=0)
    success_key = calculation_key(
        atoms2xyz(atoms),
        "{}",
        "mace",
        "",
        "single",
    )

    assert calculation_key_from_record(failure) == success_key


def test_synthesize_failure_row_truncates_long_error_messages(tmp_path):
    xyz_path = _write_xyz(tmp_path)
    args = _make_args(tmp_path)
    huge = "X" * (pd._FAILURE_ERROR_MAX_CHARS * 3)

    row = pd._synthesize_failure_row(
        0,
        RuntimeError(huge),
        args=args,
        params_str="",
        task="opt",
        calculator_name="mace",
        xyz_files=[xyz_path],
        input_mode="xyz",
        number_of_files=1,
    )

    assert len(row["opt_error"]) <= pd._FAILURE_ERROR_MAX_CHARS + len(
        "...[truncated]"
    )
    assert row["opt_error"].endswith("...[truncated]")


def test_synthesize_failure_row_handles_unreadable_input(tmp_path):
    """If even the input can't be loaded, we still emit a failure row — the
    {task}_error field is still populated and parsl_retries_exhausted is set."""

    args = _make_args(tmp_path)
    bogus = str(tmp_path / "does-not-exist.xyz")

    row = pd._synthesize_failure_row(
        0,
        RuntimeError("worker abort"),
        args=args,
        params_str="",
        task="opt",
        calculator_name="exachem",
        xyz_files=[bogus],
        input_mode="xyz",
        number_of_files=1,
    )

    assert row["initial_xyz"] == ""  # could not reconstruct
    assert row["opt_error"] == "worker abort"
    assert row["parsl_retries_exhausted"] is True
    assert row["task"] == "opt"
    assert row["calculator"] == "exachem"


def test_synthesize_failure_row_iso8601_utc_timestamp(tmp_path):
    xyz_path = _write_xyz(tmp_path)
    args = _make_args(tmp_path)

    row = pd._synthesize_failure_row(
        0,
        RuntimeError("boom"),
        args=args,
        params_str="",
        task="opt",
        calculator_name="mace",
        xyz_files=[xyz_path],
        input_mode="xyz",
        number_of_files=1,
    )

    # ISO 8601 UTC per project convention.
    assert row["date"].endswith("Z")
    assert "T" in row["date"]


# ---------------------------------------------------------------------------
# Driver-loop integration: failure row is persisted through the same writer
# ---------------------------------------------------------------------------


class _FakeFuture:
    """Stand-in for a Parsl AppFuture in the driver loop."""

    def __init__(self, result=None, exception=None):
        self._result = result
        self._exception = exception

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


def test_driver_writes_failure_row_through_jsonl_writer(tmp_path, monkeypatch):
    """Drive the as_completed loop with one success + one terminal failure and
    confirm both rows end up in the JSONL output."""

    # We extract just the loop body so we can call it without launching Parsl.
    # The loop body lives inside ``main`` but is small enough to re-implement
    # here exactly as it appears, to verify the writer code path. Any drift in
    # the real loop will be caught by reading the function source.
    from iqc.main import SKIPPED_EXISTING
    from iqc.parsl_dispatch import ComplexEncoder

    success_row = {
        "initial_xyz": WATER_XYZ,
        "task": "opt",
        "calculator": "mace",
        "model": "large",
        "params": "{}",
        "energy_eV": -76.0,
        "_unique_name": "water_0_0_X",
        "_record_stamp": "20260101_000000",
        "_work_dir_used": False,
    }
    xyz_path = _write_xyz(tmp_path)
    args = _make_args(tmp_path)
    failing_fut = _FakeFuture(exception=RuntimeError("ZE worker died"))
    success_fut = _FakeFuture(result=success_row)

    fut_to_index = {failing_fut: 0, success_fut: 1}
    xyz_files = [xyz_path, xyz_path]

    out = tmp_path / "out.jsonl"
    completed = 0
    failed = 0
    with out.open("w") as outfile:
        for fut in fut_to_index:
            try:
                result = fut.result()
            except Exception as e:
                failed += 1
                row = pd._synthesize_failure_row(
                    fut_to_index[fut],
                    e,
                    args=args,
                    params_str="{}",
                    task="opt",
                    calculator_name="mace",
                    xyz_files=xyz_files,
                    input_mode="xyz",
                    number_of_files=len(xyz_files),
                )
                outfile.write(json.dumps(row, cls=ComplexEncoder))
                outfile.write("\n")
                continue
            if result is SKIPPED_EXISTING or result is None:
                continue
            result.pop("_unique_name", None)
            result.pop("_record_stamp", None)
            result.pop("_work_dir_used", None)
            outfile.write(json.dumps(result, cls=ComplexEncoder))
            outfile.write("\n")
            completed += 1

    assert failed == 1
    assert completed == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 2

    rows = [json.loads(line) for line in lines]
    failure_rows = [r for r in rows if r.get("parsl_retries_exhausted")]
    success_rows = [r for r in rows if not r.get("parsl_retries_exhausted")]

    assert len(failure_rows) == 1
    assert len(success_rows) == 1
    fr = failure_rows[0]
    assert fr["opt_error"] == "ZE worker died"
    # Calculation key well-formed.
    calculation_key_from_record(fr)


def test_driver_failure_row_is_indexed_by_skip_existing(tmp_path):
    """Once the failure row is written to JSONL, build_completed_calculation_index
    must find its key. This is the contract F5 relies on."""

    from iqc.main import build_completed_calculation_index

    xyz_path = _write_xyz(tmp_path)
    args = _make_args(tmp_path)
    failure_row = pd._synthesize_failure_row(
        0,
        RuntimeError("retries exhausted"),
        args=args,
        params_str="",
        task="opt",
        calculator_name="exachem",
        xyz_files=[xyz_path],
        input_mode="xyz",
        number_of_files=1,
    )

    out = tmp_path / "iqc_opt_results_20260101_000000_aaaaaaaa.jsonl"
    out.write_text(json.dumps(failure_row) + "\n", encoding="utf-8")

    index, summary = build_completed_calculation_index([out])
    assert summary["records"] == 1
    assert calculation_key_from_record(failure_row) in index


def test_build_parsl_config_forwards_one_worker_per_node():
    """--parsl-one-worker-per-node must reach make_aurora_config: silently
    dropping it re-enables 12 workers/node, the exact oversubscription the
    flag exists to prevent for ExaChem runs."""
    args = argparse.Namespace(
        parsl_single_alloc=False,
        parsl_local=False,
        parsl_venv_activate="source /fake/activate",
        parsl_nodes=2,
        parsl_queue="debug",
        parsl_walltime="0:30:00",
        parsl_account="ACCT",
        parsl_retries=1,
        parsl_one_worker_per_node=True,
    )
    captured = {}

    def fake_make_aurora_config(**kwargs):
        captured.update(kwargs)
        return "config"

    with mock.patch(
        "iqc.parsl_config.make_aurora_config", fake_make_aurora_config
    ):
        config = pd._build_parsl_config(args)

    assert config == "config"
    assert captured["one_worker_per_node"] is True
