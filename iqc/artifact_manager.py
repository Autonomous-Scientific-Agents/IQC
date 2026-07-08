"""Tar/gzip ExaChem run directories so amplitudes survive scratch cleanup.

The sweep produces large per-run directories (MOs, t1/t2 amplitudes, Cholesky
vectors). Keeping them on scratch would either fill the filesystem or be wiped
by job exit; archiving them under a project-space root preserves the data for
restarts (F10) and post-hoc analysis while keeping individual run trees small
enough to compress quickly.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# Default destination on Aurora project space. Overridable via the
# ``IQC_ARTIFACT_ROOT`` environment variable or the CLI ``--artifact-root`` flag.
DEFAULT_DESTINATION_ROOT = Path(
    "/lus/flare/projects/HiFiThermKin/keceli/exachem/artifacts/"
)


class ArtifactRetentionError(RuntimeError):
    """Raised when archive creation, disk-space, or cleanup checks fail."""


def _resolve_destination_root(override: Optional[Path] = None) -> Path:
    """Return the configured archive root, honouring env override and CLI flag."""

    if override is not None:
        return Path(override)
    env = os.environ.get("IQC_ARTIFACT_ROOT")
    if env:
        return Path(env)
    return DEFAULT_DESTINATION_ROOT


def _manifest_digest(manifest: Iterable[Dict[str, Any]], run_dir: Path) -> str:
    """Stable identifier for an archive: includes run path and per-file shas.

    We want repeated calls with the same run_dir + same file contents to land
    on the same archive name so re-archiving is idempotent.
    """

    h = hashlib.sha256()
    h.update(str(run_dir.resolve()).encode("utf-8"))
    h.update(b"\0")
    # Sort by path so manifest order doesn't change the digest.
    for entry in sorted(manifest, key=lambda e: e.get("path", "")):
        h.update(str(entry.get("path", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(str(entry.get("sha256", "")).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_run_dir(
    run_dir: Path,
    manifest: List[Dict[str, Any]],
    *,
    destination_root: Optional[Path] = None,
    compress: bool = True,
) -> Dict[str, Any]:
    """Tar (+gzip) ``run_dir`` into ``destination_root`` and return a record.

    Idempotent: if an archive with the manifest-derived name already exists at
    the destination, we return its existing sha/size record instead of
    re-archiving.
    """

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ArtifactRetentionError(
            f"archive_run_dir: run_dir does not exist or is not a directory: {run_dir}"
        )

    destination_root = _resolve_destination_root(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    digest = _manifest_digest(manifest, run_dir)
    suffix = ".tar.gz" if compress else ".tar"
    archive_name = f"{run_dir.name}-{digest[:16]}{suffix}"
    archive_path = destination_root / archive_name

    if archive_path.exists():
        # Idempotent: trust the existing file. Re-hash so callers can rely on
        # the returned sha matching the on-disk contents.
        stat = archive_path.stat()
        return {
            "archive_path": str(archive_path.resolve()),
            "archive_sha256": _sha256_of_file(archive_path),
            "archive_size_bytes": stat.st_size,
            "manifest": manifest,
        }

    mode = "w:gz" if compress else "w"
    # Write to a temp file in the same dir and rename, so a crashed run doesn't
    # leave a half-written archive that future idempotent calls would trust.
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive_name}.",
        suffix=".part",
        dir=str(destination_root),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, mode) as tar:
            tar.add(str(run_dir), arcname=run_dir.name)
        tmp_path.replace(archive_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    stat = archive_path.stat()
    return {
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": _sha256_of_file(archive_path),
        "archive_size_bytes": stat.st_size,
        "manifest": manifest,
    }


def disk_space_check(
    destination_root: Path,
    projected_bytes: int,
    *,
    warn_threshold: float = 0.9,
) -> None:
    """Refuse to archive when projected usage exceeds a fraction of available bytes."""

    destination_root = Path(destination_root)
    # Walk up until we hit something that exists; statvfs needs a real path.
    probe = destination_root
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            raise ArtifactRetentionError(
                f"disk_space_check: cannot resolve any existing parent of {destination_root}"
            )
        probe = parent

    stat = os.statvfs(str(probe))
    available = stat.f_bavail * stat.f_frsize
    if available <= 0:
        raise ArtifactRetentionError(
            f"disk_space_check: filesystem reports no available bytes at {probe}"
        )
    if projected_bytes > warn_threshold * available:
        raise ArtifactRetentionError(
            f"disk_space_check: projected {projected_bytes} bytes exceeds "
            f"{warn_threshold:.0%} of {available} available at {probe}"
        )


def cleanup_by_policy(
    destination_root: Path,
    *,
    keep_n_days: int = 30,
    dry_run: bool = True,
) -> List[Path]:
    """List (and optionally remove) archives older than ``keep_n_days``."""

    destination_root = Path(destination_root)
    if not destination_root.exists():
        return []

    cutoff = datetime.datetime.now(datetime.timezone.utc).timestamp() - (
        keep_n_days * 86400
    )
    candidates: List[Path] = []
    for entry in destination_root.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        # Only archives we own; never touch arbitrary user files.
        if not (name.endswith(".tar.gz") or name.endswith(".tar")):
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            candidates.append(entry)

    if not dry_run:
        for path in candidates:
            try:
                path.unlink()
            except OSError:
                # Best-effort cleanup; leave the file in place if something
                # else holds it open.
                pass

    return candidates


__all__ = [
    "ArtifactRetentionError",
    "DEFAULT_DESTINATION_ROOT",
    "archive_run_dir",
    "cleanup_by_policy",
    "disk_space_check",
]
