"""Tests for iqc.sweep_bookkeeping (DuckDB query layer + atomic claiming)."""

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
    # Two chunks, two winners, distinct files, then nothing left.
    assert {first.name, second.name} == {"c0.parquet", "c1.parquet"}
    assert first != second
    assert third is None
    # Claimed files live under <root>/<user>/ and are gone from todo.
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
            # Simulate another user winning c0 the instant before us.
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

    # A failed chunk returns to todo/ and is picked up again.
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

    # Union the keys so every row has every column (None where absent).
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

    # New-style file: carries the stored unique_name_base column.
    _write_parquet(
        by_job / "job_a.parquet",
        [
            {
                "unique_name": "C2H6_conf0_0_0_20260101_000000",
                "unique_name_base": "C2H6_conf0",
                "total_energy_eV": -100.0,
            },
            {
                # Null energy => NOT done, even though the row exists.
                "unique_name": "C3H8_conf0_0_0_20260101_000001",
                "unique_name_base": "C3H8_conf0",
                "total_energy_eV": None,
            },
        ],
    )
    # Legacy file: no unique_name_base column -> exercises the regex fallback.
    _write_parquet(
        by_job / "job_legacy.parquet",
        [
            {
                "unique_name": "C4H10_conf0_0_0_20251231_235959",
                "total_energy_eV": -200.0,
            }
        ],
    )

    glob = str(by_job / "*.parquet")
    done = bk.done_uids(glob)
    assert done == {"C2H6_conf0", "C4H10_conf0"}  # C3H8 excluded (null energy)

    summ = bk.summary(glob)
    assert summ["total_rows"] == 3
    assert summ["distinct_uids"] == 3
    assert summ["done_uids"] == 2
    assert summ["not_done_uids"] == 1

    # Input work list keyed by the stable base identity.
    _write_parquet(
        tmp_path / "input.parquet",
        [
            {"unique_name": "C2H6_conf0"},  # done
            {"unique_name": "C3H8_conf0"},  # null energy -> still remaining
            {"unique_name": "C4H10_conf0"},  # done (via legacy fallback)
            {"unique_name": "C5H12_conf0"},  # never attempted
        ],
    )
    rem = bk.remaining(str(tmp_path / "input.parquet"), glob)
    assert rem == ["C3H8_conf0", "C5H12_conf0"]


def test_done_uids_bad_energy_column(tmp_path):
    pytest.importorskip("duckdb")
    p = tmp_path / "j.parquet"
    _write_parquet(p, [{"unique_name": "X_conf0", "total_energy_eV": -1.0}])
    with pytest.raises(ValueError, match="not found"):
        bk.done_uids(str(p), energy_column="no_such_col")
