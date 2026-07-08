"""Tests for :mod:`iqc.sweep_orchestrator`.

No tests in this file invoke a real ``qsub``/``qstat``. The PBS-binary check
on PATH gates only the (currently empty) live-integration cases below.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iqc import sweep_orchestrator as so


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, num_rows: int, with_heavy: bool = True) -> Path:
    cols = {"id": list(range(num_rows)), "xyz": [f"mol_{i}" for i in range(num_rows)]}
    if with_heavy:
        # Spread across three heavy-atom classes so chunk-sizing branches.
        cols["num_heavy_atoms"] = [3 + (i % 3) for i in range(num_rows)]
    table = pa.table(cols)
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------------
# chunk_inputs
# ---------------------------------------------------------------------------


def test_chunk_inputs_preserves_row_count(tmp_path):
    src = _write_parquet(tmp_path / "in.parquet", num_rows=1000, with_heavy=True)
    chunks = so.chunk_inputs(src, chunk_size_hint=128, output_dir=tmp_path / "out")

    assert chunks, "chunker produced no output"
    total = sum(pq.read_table(p).num_rows for p in chunks)
    assert total == 1000


def test_chunk_inputs_rows_are_non_overlapping(tmp_path):
    src = _write_parquet(tmp_path / "in.parquet", num_rows=500, with_heavy=True)
    chunks = so.chunk_inputs(src, chunk_size_hint=64, output_dir=tmp_path / "out")

    seen_ids = []
    for p in chunks:
        seen_ids.extend(pq.read_table(p).column("id").to_pylist())
    assert sorted(seen_ids) == list(range(500))
    assert len(seen_ids) == len(set(seen_ids))


def test_chunk_inputs_uniform_when_no_heavy_column(tmp_path):
    src = _write_parquet(tmp_path / "in.parquet", num_rows=300, with_heavy=False)
    chunks = so.chunk_inputs(src, chunk_size_hint=100, output_dir=tmp_path / "out")

    assert len(chunks) == 3
    for p in chunks:
        assert pq.read_table(p).num_rows == 100


def test_chunk_inputs_by_heavy_atoms_uses_per_class_sizes(tmp_path):
    # 6 rows of heavy=3, 6 rows of heavy=9. Default sizes: 3->2048 (1 chunk),
    # 9->4 (2 chunks). Total expected: 3 chunks.
    table = pa.table(
        {
            "id": list(range(12)),
            "num_heavy_atoms": [3] * 6 + [9] * 6,
        }
    )
    src = tmp_path / "mixed.parquet"
    pq.write_table(table, src)

    chunks = so.chunk_inputs(src, output_dir=tmp_path / "out")
    assert len(chunks) == 3
    sizes = [pq.read_table(p).num_rows for p in chunks]
    assert sum(sizes) == 12


def test_chunk_inputs_empty_input_returns_empty_list(tmp_path):
    src = _write_parquet(tmp_path / "empty.parquet", num_rows=0, with_heavy=True)
    assert so.chunk_inputs(src, output_dir=tmp_path / "out") == []


def test_chunk_inputs_raises_on_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        so.chunk_inputs(tmp_path / "nope.parquet")


# ---------------------------------------------------------------------------
# submit_sweep (dry-run)
# ---------------------------------------------------------------------------


def test_submit_sweep_dry_run_writes_registry_rows_without_qsub(tmp_path):
    src = _write_parquet(tmp_path / "in.parquet", num_rows=300, with_heavy=False)
    chunks = so.chunk_inputs(src, chunk_size_hint=100, output_dir=tmp_path / "out")
    registry = tmp_path / "reg.sqlite"

    with mock.patch.object(so, "_qsub") as q, mock.patch.object(
        so, "_count_user_queued_jobs", return_value=0
    ):
        so.submit_sweep(
            chunks,
            registry_db=registry,
            dry_run=True,
            script_dir=tmp_path / "scripts",
            log_dir=tmp_path / "logs",
        )
        q.assert_not_called()

    conn = sqlite3.connect(registry)
    rows = list(conn.execute("SELECT chunk_path, status FROM sweep_jobs"))
    conn.close()
    assert len(rows) == 3
    assert all(status == "would_submit" for _, status in rows)
    assert {Path(p).name for p, _ in rows} == {p.name for p in chunks}


def test_submit_sweep_skips_already_recorded_chunks(tmp_path):
    src = _write_parquet(tmp_path / "in.parquet", num_rows=200, with_heavy=False)
    chunks = so.chunk_inputs(src, chunk_size_hint=100, output_dir=tmp_path / "out")
    registry = tmp_path / "reg.sqlite"

    # First pass records all as would_submit.
    with mock.patch.object(so, "_qsub") as q, mock.patch.object(
        so, "_count_user_queued_jobs", return_value=0
    ):
        so.submit_sweep(chunks, registry_db=registry, dry_run=True)
        # Second pass should leave existing rows alone (no qsub, no errors).
        so.submit_sweep(chunks, registry_db=registry, dry_run=False)
        q.assert_not_called()


def test_submit_sweep_records_qsub_failure_and_continues(tmp_path):
    src = _write_parquet(tmp_path / "in.parquet", num_rows=200, with_heavy=False)
    chunks = so.chunk_inputs(src, chunk_size_hint=100, output_dir=tmp_path / "out")
    registry = tmp_path / "reg.sqlite"

    err = subprocess.CalledProcessError(1, ["qsub"], stderr="queue full")
    with mock.patch.object(so, "_qsub", side_effect=[err, "999.aurora"]), mock.patch.object(
        so, "_count_user_queued_jobs", return_value=0
    ), mock.patch.object(so, "_refresh_states"):
        so.submit_sweep(
            chunks,
            registry_db=registry,
            script_dir=tmp_path / "scripts",
            log_dir=tmp_path / "logs",
        )

    snap = so.status(registry)
    assert snap["counts"].get("failed", 0) == 1
    assert snap["counts"].get("queued", 0) == 1
    assert snap["failed"][0]["error"].startswith("qsub:")


def test_submit_sweep_throttles_on_queue_cap(tmp_path):
    src = _write_parquet(tmp_path / "in.parquet", num_rows=100, with_heavy=False)
    chunks = so.chunk_inputs(src, chunk_size_hint=100, output_dir=tmp_path / "out")
    registry = tmp_path / "reg.sqlite"

    queued_seq = iter([4, 4, 0])  # full, full, then free.
    sleeps: list[float] = []

    with mock.patch.object(
        so, "_count_user_queued_jobs", side_effect=lambda user=None: next(queued_seq)
    ), mock.patch.object(so, "_qsub", return_value="111.aurora"), mock.patch.object(
        so, "_refresh_states"
    ), mock.patch.object(
        so.time, "sleep", side_effect=sleeps.append
    ):
        so.submit_sweep(
            chunks,
            registry_db=registry,
            max_queued=4,
            poll_interval_s=5,
            script_dir=tmp_path / "scripts",
            log_dir=tmp_path / "logs",
        )

    assert sleeps == [5, 5]


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_counts_match_inserted_rows(tmp_path):
    registry = tmp_path / "reg.sqlite"
    conn = so._open_registry(registry)
    rows = [
        ("a.parquet", "1.aurora", "queued"),
        ("b.parquet", "2.aurora", "running"),
        ("c.parquet", "3.aurora", "completed"),
        ("d.parquet", "4.aurora", "completed"),
        ("e.parquet", None, "failed"),
    ]
    for cp, job, st in rows:
        conn.execute(
            "INSERT INTO sweep_jobs (chunk_path, pbs_job_id, status, error) "
            "VALUES (?, ?, ?, ?)",
            (cp, job, st, "boom" if st == "failed" else None),
        )
    conn.commit()
    conn.close()

    snap = so.status(registry)
    assert snap["total"] == 5
    assert snap["counts"] == {
        "queued": 1,
        "running": 1,
        "completed": 2,
        "failed": 1,
    }
    assert len(snap["failed"]) == 1
    assert snap["failed"][0]["chunk_path"] == "e.parquet"
    assert snap["failed"][0]["error"] == "boom"


# ---------------------------------------------------------------------------
# qstat parsing
# ---------------------------------------------------------------------------


def test_parse_qstat_user_count_counts_only_user_rows():
    stdout = (
        "aurora-pbs-0001:\n"
        "                                                            Req'd  Req'd   Elap\n"
        "Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n"
        "--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n"
        "123.aurora      keceli   prod     iqc_1         --   1  --     --  12:00 Q   --\n"
        "124.aurora      keceli   prod     iqc_2         --   1  --     --  12:00 R 00:05\n"
        "125.aurora      other    prod     foo           --   1  --     --  12:00 Q   --\n"
    )
    assert so._parse_qstat_user_count(stdout, "keceli") == 2
    assert so._parse_qstat_user_count(stdout, "other") == 1
    assert so._parse_qstat_user_count(stdout, "nobody") == 0


# ---------------------------------------------------------------------------
# Live PBS integration (skipped unless qsub is on PATH).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("qsub") is None, reason="qsub not on PATH; skipping live PBS test."
)
def test_count_user_queued_jobs_runs_against_real_qstat():
    # Smoke check: should not raise; returns an int.
    n = so._count_user_queued_jobs()
    assert isinstance(n, int) and n >= 0
