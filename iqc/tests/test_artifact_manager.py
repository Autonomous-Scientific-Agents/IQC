"""Tests for F9 artifact retention / compression helpers.

These tests stay fully local — no PBS, no ExaChem, no network. F4's manifest
schema is mocked here so this test file is independent of F4 landing.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import tarfile
import time
from pathlib import Path

import pytest

from iqc.artifact_manager import (
    ArtifactRetentionError,
    archive_run_dir,
    cleanup_by_policy,
    disk_space_check,
)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _make_run_dir(tmp_path: Path) -> tuple[Path, list[dict]]:
    """Build a fake ExaChem run_dir with a manifest matching F4's contract."""

    run_dir = tmp_path / "exachem_run"
    run_dir.mkdir()
    files = {
        "input.json": b'{"task": "scf"}',
        "h2o.cc-pvdz_files/restricted/h2o.mo": b"mo-coeffs-here",
        "h2o.cc-pvdz_files/restricted/h2o.t1amp": b"t1-amplitudes",
        "exachem.log": b"...converged...",
    }
    manifest = []
    for rel, data in files.items():
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        kind = "output" if rel.endswith(".log") or rel.endswith(".json") else (
            "mo" if rel.endswith(".mo") else "amplitudes"
        )
        manifest.append(
            {
                "path": str(path.resolve()),
                "kind": kind,
                "size_bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
    return run_dir, manifest


def test_archive_run_dir_creates_targz_and_returns_record(tmp_path):
    run_dir, manifest = _make_run_dir(tmp_path)
    dest = tmp_path / "archives"

    record = archive_run_dir(run_dir, manifest, destination_root=dest)

    archive_path = Path(record["archive_path"])
    assert archive_path.exists()
    assert archive_path.suffix == ".gz"
    assert archive_path.name.endswith(".tar.gz")
    assert archive_path.name.startswith("exachem_run-")
    assert record["archive_size_bytes"] == archive_path.stat().st_size
    assert record["archive_sha256"] == hashlib.sha256(
        archive_path.read_bytes()
    ).hexdigest()
    assert record["manifest"] == manifest

    # Archive is a real tarball containing the source files.
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
    assert "exachem_run/input.json" in names
    assert any(n.endswith("h2o.mo") for n in names)
    assert any(n.endswith("h2o.t1amp") for n in names)


def test_archive_run_dir_is_idempotent(tmp_path):
    run_dir, manifest = _make_run_dir(tmp_path)
    dest = tmp_path / "archives"

    first = archive_run_dir(run_dir, manifest, destination_root=dest)
    archive_path = Path(first["archive_path"])
    original_mtime = archive_path.stat().st_mtime
    original_size = archive_path.stat().st_size

    # Sleep just enough that any rewrite would bump mtime measurably.
    time.sleep(0.05)

    second = archive_run_dir(run_dir, manifest, destination_root=dest)

    assert second["archive_path"] == first["archive_path"]
    assert second["archive_sha256"] == first["archive_sha256"]
    assert second["archive_size_bytes"] == first["archive_size_bytes"]
    # mtime unchanged — we didn't re-tar.
    assert archive_path.stat().st_mtime == original_mtime
    assert archive_path.stat().st_size == original_size


def test_archive_run_dir_uncompressed_writes_plain_tar(tmp_path):
    run_dir, manifest = _make_run_dir(tmp_path)
    dest = tmp_path / "archives"

    record = archive_run_dir(
        run_dir, manifest, destination_root=dest, compress=False
    )
    archive_path = Path(record["archive_path"])
    assert archive_path.name.endswith(".tar")
    assert not archive_path.name.endswith(".tar.gz")
    with tarfile.open(archive_path, "r:") as tar:
        names = tar.getnames()
    assert "exachem_run/input.json" in names


def test_archive_run_dir_rejects_missing_dir(tmp_path):
    with pytest.raises(ArtifactRetentionError):
        archive_run_dir(
            tmp_path / "does-not-exist",
            [],
            destination_root=tmp_path / "archives",
        )


def test_disk_space_check_raises_when_projection_exceeds_available(tmp_path):
    # f_bavail * f_frsize for any real dir is finite; projecting 10x available
    # should always trip the threshold.
    stat = os.statvfs(str(tmp_path))
    available = stat.f_bavail * stat.f_frsize
    if available == 0:
        pytest.skip("Filesystem reports zero available bytes")

    with pytest.raises(ArtifactRetentionError):
        disk_space_check(tmp_path, projected_bytes=available * 10)


def test_disk_space_check_passes_for_tiny_projection(tmp_path):
    # Projecting 1 byte should never trip the 90% threshold on a non-full FS.
    disk_space_check(tmp_path, projected_bytes=1)


def test_disk_space_check_walks_up_to_existing_parent(tmp_path):
    # Destination that doesn't exist yet should still resolve to a parent.
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    disk_space_check(nested, projected_bytes=1)


def test_cleanup_by_policy_dry_run_lists_old_archives_without_removing(tmp_path):
    dest = tmp_path / "archives"
    dest.mkdir()

    fresh = dest / "fresh.tar.gz"
    stale = dest / "stale.tar.gz"
    other = dest / "not_an_archive.txt"
    fresh.write_bytes(b"new")
    stale.write_bytes(b"old")
    other.write_bytes(b"keep me")

    # Backdate the stale one by 45 days.
    old_ts = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=45)
    ).timestamp()
    os.utime(stale, (old_ts, old_ts))

    candidates = cleanup_by_policy(dest, keep_n_days=30, dry_run=True)

    assert stale in candidates
    assert fresh not in candidates
    assert other not in candidates
    # dry_run must not delete anything.
    assert stale.exists()
    assert fresh.exists()
    assert other.exists()


def test_cleanup_by_policy_removes_when_dry_run_false(tmp_path):
    dest = tmp_path / "archives"
    dest.mkdir()
    stale = dest / "stale.tar.gz"
    stale.write_bytes(b"old")
    old_ts = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=45)
    ).timestamp()
    os.utime(stale, (old_ts, old_ts))

    removed = cleanup_by_policy(dest, keep_n_days=30, dry_run=False)

    assert stale in removed
    assert not stale.exists()


def test_cleanup_by_policy_returns_empty_when_destination_missing(tmp_path):
    assert cleanup_by_policy(tmp_path / "nowhere", keep_n_days=30) == []


def test_iqc_artifact_root_env_var_overrides_default(monkeypatch, tmp_path):
    """When destination_root is None, IQC_ARTIFACT_ROOT controls placement."""

    run_dir, manifest = _make_run_dir(tmp_path)
    env_root = tmp_path / "from-env"
    monkeypatch.setenv("IQC_ARTIFACT_ROOT", str(env_root))

    record = archive_run_dir(run_dir, manifest)

    assert Path(record["archive_path"]).parent == env_root.resolve()
