"""Tests for iqc.sweep_bookkeeping (DuckDB query layer + atomic claiming)
and the producer-side identity that feeds it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from iqc import sweep_bookkeeping as bk


# --------------------------------------------------------------------------- #
# Identity fallback (no DuckDB needed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "full,base",
    [
        ("C2H6N2O_conf0077_0_0_20260630_164634", "C2H6N2O_conf0077"),
        ("C10H22_conf0000_12_3_20260101_000000", "C10H22_conf0000"),
        ("already_base_no_suffix", "already_base_no_suffix"),  # idempotent
    ],
)
def test_base_from_unique_name(full, base):
    assert bk.base_from_unique_name(full) == base


# --------------------------------------------------------------------------- #
# Producer identity: compute_unique_name_base + tabular reader UID (P1)
# --------------------------------------------------------------------------- #


class _Rec:
    def __init__(self, uid):
        self.uid = uid


def test_compute_unique_name_base_uses_input_uid_for_tabular():
    from iqc.main import compute_unique_name_base

    recs = [_Rec("H2_conf0000"), _Rec("H2_conf0001")]
    # Tabular: carry the input UID, not the {stem}_row{i} filename base.
    assert compute_unique_name_base("data_xyz", recs, 1, 2, 2, "chunk_001_row1") == "H2_conf0001"
    # No UID on the record -> fall back to base_name.
    assert compute_unique_name_base("data_xyz", [_Rec(None)], 0, 1, 1, "chunk_row0") == "chunk_row0"


def test_compute_unique_name_base_keeps_frames_distinct():
    from iqc.main import compute_unique_name_base

    # Multi-frame single file: each frame gets a distinct identity.
    b0 = compute_unique_name_base("xyz", ["f.xyz"], 0, 1, 3, "traj")
    b1 = compute_unique_name_base("xyz", ["f.xyz"], 1, 1, 3, "traj")
    assert b0 == "traj_frame0" and b1 == "traj_frame1"
    # A directory of files (number_of_files > 1) keeps the per-file base.
    assert compute_unique_name_base("xyz", ["a", "b"], 0, 2, 2, "molA") == "molA"
    # A single-frame single file is unchanged.
    assert compute_unique_name_base("xyz", ["f.xyz"], 0, 1, 1, "molA") == "molA"


def test_reader_attaches_uid_from_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from iqc.datatools import read_xyz_column_records

    p = tmp_path / "chunk.parquet"
    pq.write_table(
        pa.table({"opt_xyz": ["1\n\nH 0 0 0", "1\n\nH 0 0 1"],
                  "unique_name": ["H2_conf0000", "H2_conf0001"]}),
        str(p),
    )
    recs = read_xyz_column_records(str(p), "opt_xyz", uid_column="unique_name")
    assert [r.uid for r in recs] == ["H2_conf0000", "H2_conf0001"]

    # Auto default (uid_required=False): a missing column is tolerated.
    recs2 = read_xyz_column_records(str(p), "opt_xyz", uid_column="nope", uid_required=False)
    assert [r.uid for r in recs2] == [None, None]
    # Explicit request for a missing column raises.
    with pytest.raises(Exception):
        read_xyz_column_records(str(p), "opt_xyz", uid_column="nope", uid_required=True)


# --------------------------------------------------------------------------- #
# Atomic chunk claiming (pure os.rename — no DuckDB needed)
# --------------------------------------------------------------------------- #


def _make_chunks(todo: Path, names):
    todo.mkdir(parents=True, exist_ok=True)
    for n in names:
        (todo / n).write_text("x")


def test_claim_is_exactly_once(tmp_path):
    todo = tmp_path / "todo"
    claimed = tmp_path / "claimed"
    _make_chunks(todo, ["c0.parquet", "c1.parquet"])

    first = bk.claim_chunk(todo, claimed, "alice")
    second = bk.claim_chunk(todo, claimed, "bob")
    third = bk.claim_chunk(todo, claimed, "carol")

    assert first is not None and second is not None
    assert {first.name, second.name} == {"c0.parquet", "c1.parquet"}
    assert first != second
    assert third is None
    assert first.parent.name == "alice"
    assert second.parent.name == "bob"
    assert list(todo.glob("*.parquet")) == []


def test_claim_returns_none_when_empty(tmp_path):
    todo = tmp_path / "todo"
    todo.mkdir()
    assert bk.claim_chunk(todo, tmp_path / "claimed", "alice") is None


def test_claim_survives_a_lost_race(tmp_path, monkeypatch):
    """If a candidate vanishes mid-claim, the claimer moves to the next one."""
    todo = tmp_path / "todo"
    claimed = tmp_path / "claimed"
    _make_chunks(todo, ["c0.parquet", "c1.parquet"])

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError(src)
        return real_rename(src, dst)

    monkeypatch.setattr(bk.os, "rename", flaky_rename)
    got = bk.claim_chunk(todo, claimed, "alice")
    assert got is not None
    assert got.name == "c1.parquet"


def test_complete_and_fail_move_chunks(tmp_path):
    todo = tmp_path / "todo"
    done = tmp_path / "done"
    claimed = tmp_path / "claimed" / "alice"
    claimed.mkdir(parents=True)
    chunk = claimed / "c0.parquet"
    chunk.write_text("x")

    moved = bk.complete_chunk(chunk, done)
    assert moved == done / "c0.parquet"
    assert moved.exists() and not chunk.exists()

    failed = bk.fail_chunk(moved, todo)
    assert failed == todo / "c0.parquet"
    assert failed.exists()
    assert bk.claim_chunk(todo, tmp_path / "claimed", "bob").name == "c0.parquet"


# --------------------------------------------------------------------------- #
# DuckDB query layer (skipped if duckdb is not installed)
# --------------------------------------------------------------------------- #


def _write_parquet(path: Path, rows: list[dict]):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    cols = {k: [r.get(k) for r in rows] for k in keys}
    pq.write_table(pa.table(cols), str(path))


def test_done_and_remaining_with_mixed_schema(tmp_path):
    pytest.importorskip("duckdb")

    by_job = tmp_path / "by_job"
    by_job.mkdir()

    _write_parquet(
        by_job / "job_a.parquet",
        [
            {"unique_name": "C2H6_conf0_0_0_20260101_000000",
             "unique_name_base": "C2H6_conf0", "total_energy_eV": -100.0},
            {"unique_name": "C3H8_conf0_0_0_20260101_000001",
             "unique_name_base": "C3H8_conf0", "total_energy_eV": None},  # null -> not done
        ],
    )
    # Legacy file: no unique_name_base column -> regex fallback.
    _write_parquet(
        by_job / "job_legacy.parquet",
        [{"unique_name": "C4H10_conf0_0_0_20251231_235959", "total_energy_eV": -200.0}],
    )

    glob = str(by_job / "*.parquet")
    assert bk.done_uids(glob) == {"C2H6_conf0", "C4H10_conf0"}

    summ = bk.summary(glob)
    assert summ == {"total_rows": 3, "distinct_uids": 3, "done_uids": 2, "not_done_uids": 1}

    _write_parquet(
        tmp_path / "input.parquet",
        [{"unique_name": u} for u in
         ["C2H6_conf0", "C3H8_conf0", "C4H10_conf0", "C5H12_conf0"]],
    )
    rem = bk.remaining(str(tmp_path / "input.parquet"), glob)
    assert rem == ["C3H8_conf0", "C5H12_conf0"]


def test_done_excludes_failed_and_nonphysical(tmp_path):
    """A non-null energy is not enough: failures/nonphysical/non-converged are
    not done (matches status_query)."""
    pytest.importorskip("duckdb")
    p = tmp_path / "job.parquet"
    _write_parquet(
        p,
        [
            {"unique_name_base": "ok", "total_energy_eV": -1.0},
            {"unique_name_base": "err", "total_energy_eV": -1.0, "error": "boom"},
            {"unique_name_base": "task_err", "total_energy_eV": -1.0, "single_error": "x"},
            {"unique_name_base": "nonphys", "total_energy_eV": -1.0, "nonphysical": True},
            {"unique_name_base": "noconv", "total_energy_eV": -1.0, "opt_converged": False},
            {"unique_name_base": "nan", "total_energy_eV": float("nan")},
        ],
    )
    assert bk.done_uids(str(p)) == {"ok"}


def test_reads_interrupted_jsonl(tmp_path):
    """A walltime-killed job leaves JSONL with a truncated last line; valid
    rows before it must still be read (P2)."""
    pytest.importorskip("duckdb")
    f = tmp_path / "iqc_single_results_run.jsonl"
    f.write_text(
        '{"unique_name_base": "good", "total_energy_eV": -5.0}\n'
        '{"unique_name_base": "half", "total_energy_eV": -6.'  # truncated, no newline
    )
    assert bk.done_uids(str(tmp_path / "*.jsonl")) == {"good"}


def test_empty_result_glob_is_empty_done(tmp_path):
    """Fresh sweep, before any result file exists: remaining = all inputs,
    done = empty, summary = zeros (no IOException) (P2)."""
    pytest.importorskip("duckdb")
    by_job = tmp_path / "by_job"
    by_job.mkdir()
    glob = str(by_job / "*.parquet")

    _write_parquet(
        tmp_path / "input.parquet",
        [{"unique_name": u} for u in ["a", "b", "c"]],
    )

    assert bk.done_uids(glob) == set()
    assert bk.summary(glob) == {"total_rows": 0, "distinct_uids": 0,
                                "done_uids": 0, "not_done_uids": 0}
    assert bk.remaining(str(tmp_path / "input.parquet"), glob) == ["a", "b", "c"]


def test_base_only_schema_does_not_reference_unique_name(tmp_path):
    """A parquet with only unique_name_base (+ energy) must not fail with a
    binder error referencing the absent unique_name column (P2)."""
    pytest.importorskip("duckdb")
    p = tmp_path / "job.parquet"
    _write_parquet(p, [{"unique_name_base": "X_conf0", "total_energy_eV": -1.0}])
    assert bk.done_uids(str(p)) == {"X_conf0"}


def test_done_uids_bad_energy_column(tmp_path):
    """An explicitly requested, misspelled energy column still raises (distinct
    from a failure-only source that merely lacks the default energy column)."""
    pytest.importorskip("duckdb")
    p = tmp_path / "j.parquet"
    _write_parquet(p, [{"unique_name": "X_conf0", "total_energy_eV": -1.0}])
    with pytest.raises(ValueError, match="not found"):
        bk.done_uids(str(p), energy_column="no_such_col")


def test_corrupt_parquet_still_raises(tmp_path):
    """A genuinely unreadable file is not silently treated as 'no done'."""
    pytest.importorskip("duckdb")
    p = tmp_path / "corrupt.parquet"
    p.write_text("this is not a parquet file")
    with pytest.raises(Exception):
        bk.done_uids(str(p))


# --- failure-only / interrupted sources (P2 follow-up) --------------------- #


def test_worker_failure_row_source_is_no_done(tmp_path):
    """A real EL _synthesize_failure_row (no identity, no energy columns) must
    contribute no done UIDs, leaving its input eligible for retry."""
    pytest.importorskip("duckdb")
    import types

    from iqc.ensemble_launcher_dispatch import _synthesize_failure_row

    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0\n")
    args = types.SimpleNamespace(
        input=None, xyz=str(xyz), smiles=None, sort=None, sort_order="up",
        multiplicity=None, charge=None,
    )
    row = _synthesize_failure_row(
        0, RuntimeError("backend crashed"),
        args=args, params_str="", task="single", calculator_name="exachem",
        xyz_files=[str(xyz)], input_mode="xyz", number_of_files=1,
    )
    assert "unique_name" not in row and "unique_name_base" not in row
    assert "total_energy_eV" not in row

    by_job = tmp_path / "by_job"
    by_job.mkdir()
    _write_parquet(by_job / "iqc_single_results_run.parquet", [row])
    glob = str(by_job / "*.parquet")
    assert bk.done_uids(glob) == set()
    # summary reports the row; no identity column -> distinct/done are 0.
    assert bk.summary(glob) == {"total_rows": 1, "distinct_uids": 0,
                                "done_uids": 0, "not_done_uids": 0}

    _write_parquet(tmp_path / "input.parquet", [{"unique_name": u} for u in ["m0", "m1"]])
    assert bk.remaining(str(tmp_path / "input.parquet"), glob) == ["m0", "m1"]


def test_backend_error_row_with_identity_no_energy(tmp_path):
    """A _process_one_row result whose task raised before an energy: it has
    unique_name_base + {task}_error but no energy column. Not done; retryable."""
    pytest.importorskip("duckdb")
    by_job = tmp_path / "by_job"
    by_job.mkdir()
    _write_parquet(
        by_job / "job.parquet",
        [{"unique_name": "M_conf0_0_0_20260101_000000",
          "unique_name_base": "M_conf0", "task": "single",
          "single_error": "SCF did not converge"}],
    )
    glob = str(by_job / "*.parquet")
    assert bk.done_uids(glob) == set()
    assert bk.summary(glob) == {"total_rows": 1, "distinct_uids": 1,
                                "done_uids": 0, "not_done_uids": 1}
    _write_parquet(tmp_path / "input.parquet", [{"unique_name": "M_conf0"}])
    assert bk.remaining(str(tmp_path / "input.parquet"), glob) == ["M_conf0"]


def test_empty_jsonl_file_is_no_done(tmp_path):
    """A JSONL file opened before the first completion (empty) is not an error."""
    pytest.importorskip("duckdb")
    by_job = tmp_path / "by_job"
    by_job.mkdir()
    (by_job / "iqc_single_results_run.jsonl").write_text("")
    glob = str(by_job / "*.jsonl")
    assert bk.done_uids(glob) == set()
    assert bk.summary(glob)["done_uids"] == 0
    _write_parquet(tmp_path / "input.parquet", [{"unique_name": "z0"}])
    assert bk.remaining(str(tmp_path / "input.parquet"), glob) == ["z0"]


def test_mixed_success_and_failure_only_files(tmp_path):
    """A glob mixing a successful file and a failure-only file: done = the
    successes; the failure-only inputs remain."""
    pytest.importorskip("duckdb")
    by_job = tmp_path / "by_job"
    by_job.mkdir()
    _write_parquet(
        by_job / "ok.parquet",
        [{"unique_name_base": "good", "total_energy_eV": -1.0}],
    )
    _write_parquet(
        by_job / "fail.parquet",
        [{"unique_name_base": "bad", "single_error": "boom"}],  # no energy col here
    )
    glob = str(by_job / "*.parquet")
    # union_by_name fills total_energy_eV=NULL for the failure row.
    assert bk.done_uids(glob) == {"good"}
    _write_parquet(tmp_path / "input.parquet",
                   [{"unique_name": u} for u in ["good", "bad"]])
    assert bk.remaining(str(tmp_path / "input.parquet"), glob) == ["bad"]
