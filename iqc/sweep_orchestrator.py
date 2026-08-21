"""Sweep orchestrator: chunk a large parquet and submit ``iqc-parsl`` jobs
through PBS while respecting Aurora's 4-queued-job per-user cap.

This module exists because ``iqc-parsl`` is one PBS job per invocation
(``max_blocks=1``) and Aurora's scheduler refuses a 5th queued job from the
same user. A 116k-structure CCSD(T) sweep therefore needs many submissions
chained behind the cap; doing that by hand is error-prone and wastes hours
of wall while the user babysits ``qsub`` + ``qstat``.

What lives here:

- ``chunk_inputs``: split an input parquet into smaller chunk parquets sized
  to fit one 12 h PBS wall, optionally cost-aware by heavy-atom count
  (small molecules pack many rows per chunk, large molecules few).
- ``submit_sweep``: feed chunks into PBS, throttling on ``qstat`` so the
  user never exceeds ``max_queued`` queued jobs at once. Tracks every
  submission in a SQLite registry so a re-run can resume cleanly.
- ``status``: report registry counts and surface failed chunks.

Intentional choices: stdlib only (sqlite3, subprocess, argparse), no Parsl
dependency here — this module orchestrates ``iqc-parsl`` from outside, it
does not run inside one.
"""
from __future__ import annotations

import logging
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk sizing
# ---------------------------------------------------------------------------

# Cost-aware default chunk size per heavy-atom class. Values reflect the
# per-structure CCSD(T)/aug-cc-pVTZ cost from COST_ESTIMATE.md combined with
# the 12 h PBS wall cap: small molecules can pack thousands of rows per PBS
# job; ~C9/C10 structures may only fit a handful.
DEFAULT_HEAVY_ATOM_CHUNK_SIZES: dict[int, int] = {
    0: 4096,
    1: 4096,
    2: 4096,
    3: 2048,
    4: 1024,
    5: 512,
    6: 256,
    7: 64,
    8: 16,
    9: 4,
    10: 1,
}

# Fallback when by_heavy_atoms is disabled or the input has no heavy-atom column.
DEFAULT_FLAT_CHUNK_SIZE = 256

# Common candidate column names used by the IQC parquet schema for the
# heavy-atom count. Detected case-insensitively.
HEAVY_ATOM_COLUMN_CANDIDATES: tuple[str, ...] = (
    "num_heavy_atoms",
    "n_heavy_atoms",
    "heavy_atoms",
    "n_heavy",
)


def _chunk_size_for_heavy_atoms(heavy: int, hint: Optional[int]) -> int:
    """Pick the per-chunk row count for a given heavy-atom class."""

    if hint is not None and hint > 0:
        return hint
    if heavy in DEFAULT_HEAVY_ATOM_CHUNK_SIZES:
        return DEFAULT_HEAVY_ATOM_CHUNK_SIZES[heavy]
    # Larger than what we have data for — be conservative.
    return 1


def _detect_heavy_atom_column(schema: pa.Schema) -> Optional[str]:
    lower = {name.lower(): name for name in schema.names}
    for candidate in HEAVY_ATOM_COLUMN_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    return None


def chunk_inputs(
    input_parquet: Path,
    chunk_size_hint: Optional[int] = None,
    by_heavy_atoms: bool = True,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Split ``input_parquet`` into smaller chunk parquets sized for one PBS job.

    With ``by_heavy_atoms=True`` (default) the rows are grouped by heavy-atom
    count and each class is split with a per-class chunk size — small
    molecules pack many rows per PBS job, large molecules few. Falls back to
    a uniform chunk size if the heavy-atom column is missing or the caller
    sets ``by_heavy_atoms=False``.

    Returns the list of chunk parquet paths in submission order (heavy-atom
    ascending — cheap chunks first so the queue starts producing results
    early). Chunks are written to ``output_dir`` (created if needed),
    defaulting to a sibling directory ``<input>.chunks/``.
    """

    input_parquet = Path(input_parquet).resolve()
    if not input_parquet.is_file():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")

    if output_dir is None:
        output_dir = input_parquet.with_name(input_parquet.stem + ".chunks")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(input_parquet)
    if table.num_rows == 0:
        logger.warning("Input parquet %s has 0 rows; no chunks produced.", input_parquet)
        return []

    heavy_col = _detect_heavy_atom_column(table.schema) if by_heavy_atoms else None
    chunks: list[Path] = []

    if heavy_col is None:
        # Uniform chunking.
        chunk_size = chunk_size_hint or DEFAULT_FLAT_CHUNK_SIZE
        chunks.extend(
            _write_chunks(table, output_dir, input_parquet.stem, chunk_size, tag="all")
        )
        logger.info(
            "Chunked %s rows from %s into %s uniform chunk(s) of size %s.",
            table.num_rows,
            input_parquet.name,
            len(chunks),
            chunk_size,
        )
        return chunks

    # Group by heavy-atom count: build a per-class slice, then chunk each
    # slice with its class-specific size. PyArrow lacks groupby slicing
    # directly, so we sort by the heavy column once and walk run-length
    # boundaries to slice cheaply.
    indices = pa.compute.sort_indices(table, sort_keys=[(heavy_col, "ascending")])
    sorted_table = table.take(indices)
    heavy_values = sorted_table.column(heavy_col).to_pylist()

    start = 0
    n = sorted_table.num_rows
    while start < n:
        current = heavy_values[start]
        end = start + 1
        while end < n and heavy_values[end] == current:
            end += 1
        class_slice = sorted_table.slice(start, end - start)
        heavy_int = int(current) if current is not None else -1
        chunk_size = _chunk_size_for_heavy_atoms(heavy_int, chunk_size_hint)
        tag = f"h{heavy_int:02d}" if heavy_int >= 0 else "hNA"
        chunks.extend(
            _write_chunks(class_slice, output_dir, input_parquet.stem, chunk_size, tag=tag)
        )
        start = end

    logger.info(
        "Chunked %s rows from %s into %s heavy-atom-aware chunk(s).",
        table.num_rows,
        input_parquet.name,
        len(chunks),
    )
    return chunks


def _write_chunks(
    table: pa.Table,
    output_dir: Path,
    stem: str,
    chunk_size: int,
    tag: str,
) -> list[Path]:
    """Slice ``table`` into pieces of at most ``chunk_size`` rows and write each."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >0, got {chunk_size}")
    out: list[Path] = []
    n = table.num_rows
    if n == 0:
        return out
    nparts = (n + chunk_size - 1) // chunk_size
    width = max(4, len(str(nparts)))
    for i in range(nparts):
        slice_ = table.slice(i * chunk_size, chunk_size)
        path = output_dir / f"{stem}__{tag}__{i:0{width}d}.parquet"
        pq.write_table(slice_, path)
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# Registry (SQLite)
# ---------------------------------------------------------------------------

_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS sweep_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_path  TEXT NOT NULL,
    pbs_job_id  TEXT,
    status      TEXT NOT NULL,
    submit_time TEXT,
    finish_time TEXT,
    error       TEXT,
    UNIQUE(chunk_path)
);
"""


def _open_registry(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_REGISTRY_SCHEMA)
    return conn


def _registry_chunk_status(conn: sqlite3.Connection, chunk_path: str) -> Optional[str]:
    row = conn.execute(
        "SELECT status FROM sweep_jobs WHERE chunk_path = ?", (chunk_path,)
    ).fetchone()
    return row["status"] if row else None


def _upsert_chunk_row(
    conn: sqlite3.Connection,
    *,
    chunk_path: str,
    pbs_job_id: Optional[str],
    status: str,
    submit_time: Optional[str] = None,
    finish_time: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sweep_jobs (chunk_path, pbs_job_id, status, submit_time, finish_time, error)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_path) DO UPDATE SET
            pbs_job_id  = COALESCE(excluded.pbs_job_id, sweep_jobs.pbs_job_id),
            status      = excluded.status,
            submit_time = COALESCE(excluded.submit_time, sweep_jobs.submit_time),
            finish_time = COALESCE(excluded.finish_time, sweep_jobs.finish_time),
            error       = COALESCE(excluded.error, sweep_jobs.error)
        """,
        (chunk_path, pbs_job_id, status, submit_time, finish_time, error),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# PBS interaction
# ---------------------------------------------------------------------------


def _count_user_queued_jobs(user: Optional[str] = None) -> int:
    """Return the count of this user's jobs in Q (queued) state.

    Aurora's per-user cap counts queued + held + running together as
    "in-system", but the cap that triggers a ``qsub`` rejection is the
    queued count. We conservatively count anything not in S/E (suspended/exiting)
    state — that mirrors what ``qstat -u`` reports as user-visible jobs.
    """

    user = user or os.environ.get("USER", "")
    if not user:
        return 0
    try:
        proc = subprocess.run(
            ["qstat", "-u", user],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        # qstat not on PATH — we can't poll; treat as 0 so submissions proceed.
        logger.warning("qstat not found on PATH; queue cap polling disabled.")
        return 0
    except subprocess.TimeoutExpired:
        logger.warning("qstat timed out; assuming queue is full to back off.")
        return 9_999
    if proc.returncode != 0:
        logger.warning("qstat returned %s: %s", proc.returncode, proc.stderr.strip())
        return 0
    return _parse_qstat_user_count(proc.stdout, user)


def _parse_qstat_user_count(stdout: str, user: str) -> int:
    """Count the user's job rows in ``qstat -u <user>`` output.

    PBSPro's ``qstat -u`` output starts with header lines (server banner,
    column titles, a dashes separator) and then one row per job. We just
    count lines whose first whitespace-delimited token looks like a PBS job
    ID (digits.servername) and whose user column matches.
    """

    count = 0
    for line in stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("-"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        job_id = parts[0]
        # PBS job IDs look like 12345.aurora-pbs-0001 or 12345[].something.
        if not job_id[:1].isdigit():
            continue
        # Username is the 2nd field in the standard qstat -u layout.
        if parts[1] == user:
            count += 1
    return count


def _query_job_state(pbs_job_id: str) -> Optional[str]:
    """Return ``qstat`` state letter for a job, or None if not found.

    We try the simple ``qstat <id>`` parse first (works on any PBS), and
    fall back to absent (job is no longer in scheduler memory — treat as
    completed by caller).
    """

    try:
        proc = subprocess.run(
            ["qstat", pbs_job_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts or not parts[0][:1].isdigit():
            continue
        if parts[0].split(".")[0] == pbs_job_id.split(".")[0] and len(parts) >= 5:
            # Column 5 is the state letter in standard qstat output.
            return parts[4]
    return None


# ---------------------------------------------------------------------------
# PBS script building + submit
# ---------------------------------------------------------------------------


def _build_pbs_script(
    *,
    chunk_path: Path,
    iqc_parsl_args: Sequence[str],
    queue: str,
    walltime: str,
    account: str,
    nodes: int,
    filesystems: str,
    venv_activate: Optional[str],
    job_name: str,
    log_dir: Path,
) -> str:
    """Render a PBS script that runs ``iqc-parsl`` against ``chunk_path``."""

    venv_activate = venv_activate or os.path.join(sys.prefix, "bin", "activate")
    extra = " ".join(shlex.quote(a) for a in iqc_parsl_args)
    chunk_quoted = shlex.quote(str(chunk_path))
    return f"""#!/bin/bash -l
#PBS -N {job_name}
#PBS -l select={nodes}:system=aurora
#PBS -l walltime={walltime}
#PBS -q {queue}
#PBS -A {account}
#PBS -l filesystems={filesystems}
#PBS -o {log_dir}/{job_name}.out
#PBS -e {log_dir}/{job_name}.err

set -euo pipefail
cd "$PBS_O_WORKDIR"

export TMPDIR=/tmp
source {shlex.quote(venv_activate)}

iqc-parsl --parsl-single-alloc --input {chunk_quoted} {extra}
"""


def _qsub(script_path: Path) -> str:
    """Submit a PBS script and return the assigned job ID."""

    proc = subprocess.run(
        ["qsub", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.stdout.strip().splitlines()[-1].strip()


# ---------------------------------------------------------------------------
# Public submit/status API
# ---------------------------------------------------------------------------


def submit_sweep(
    chunk_paths: Iterable[Path],
    *,
    registry_db: Path,
    max_queued: int = 4,
    poll_interval_s: int = 60,
    iqc_parsl_args: Optional[list[str]] = None,
    dry_run: bool = False,
    queue: str = "prod",
    walltime: str = "12:00:00",
    account: str = "IQC",
    nodes: int = 1,
    filesystems: str = "home:flare",
    venv_activate: Optional[str] = None,
    script_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    job_name_prefix: str = "iqc_sweep",
    user: Optional[str] = None,
) -> None:
    """Submit ``iqc-parsl`` PBS jobs for each chunk, honoring the queue cap.

    The cap is checked via ``qstat -u $USER`` before each submission; when
    the user already owns ``max_queued`` jobs the orchestrator sleeps
    ``poll_interval_s`` seconds and re-checks. Each submission is recorded
    in ``registry_db`` so re-running ``submit_sweep`` on the same chunk list
    is idempotent — chunks already in non-failed states are skipped.

    On ``dry_run=True`` no PBS commands run: rows are written with status
    ``would_submit`` so tests and rehearsals can exercise the registry
    without touching the scheduler.
    """

    chunk_paths = [Path(p).resolve() for p in chunk_paths]
    iqc_parsl_args = list(iqc_parsl_args or [])

    script_dir = Path(script_dir).resolve() if script_dir else Path("pbs_scripts").resolve()
    log_dir = Path(log_dir).resolve() if log_dir else Path("pbs_logs").resolve()
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    conn = _open_registry(Path(registry_db))
    try:
        for chunk_path in chunk_paths:
            cp_str = str(chunk_path)
            existing = _registry_chunk_status(conn, cp_str)
            # would_submit rows come from a --dry-run rehearsal and carry no
            # PBS job; only a dry run may treat them as already handled —
            # otherwise the rehearsal permanently blocks the real submission.
            skip_states = {"queued", "running", "completed"}
            if dry_run:
                skip_states.add("would_submit")
            if existing in skip_states:
                logger.info("Skipping %s — already recorded as %s.", chunk_path.name, existing)
                continue

            if dry_run:
                _upsert_chunk_row(
                    conn,
                    chunk_path=cp_str,
                    pbs_job_id=None,
                    status="would_submit",
                    submit_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                logger.info("[dry-run] would submit %s", chunk_path.name)
                continue

            # Throttle on the queue cap.
            while True:
                queued = _count_user_queued_jobs(user=user)
                if queued < max_queued:
                    break
                logger.info(
                    "Queue cap reached (%s of %s); sleeping %ss before re-check.",
                    queued,
                    max_queued,
                    poll_interval_s,
                )
                time.sleep(poll_interval_s)

            # Build + submit.
            job_name = f"{job_name_prefix}_{chunk_path.stem}"[:64]
            script_path = script_dir / f"{job_name}.pbs"
            script_body = _build_pbs_script(
                chunk_path=chunk_path,
                iqc_parsl_args=iqc_parsl_args,
                queue=queue,
                walltime=walltime,
                account=account,
                nodes=nodes,
                filesystems=filesystems,
                venv_activate=venv_activate,
                job_name=job_name,
                log_dir=log_dir,
            )
            script_path.write_text(script_body)
            try:
                job_id = _qsub(script_path)
            except subprocess.CalledProcessError as exc:
                err = (exc.stderr or "").strip() or str(exc)
                logger.error("qsub failed for %s: %s", chunk_path.name, err)
                _upsert_chunk_row(
                    conn,
                    chunk_path=cp_str,
                    pbs_job_id=None,
                    status="failed",
                    submit_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    error=f"qsub: {err}",
                )
                continue
            except FileNotFoundError:
                logger.error("qsub not on PATH — aborting sweep submission loop.")
                _upsert_chunk_row(
                    conn,
                    chunk_path=cp_str,
                    pbs_job_id=None,
                    status="failed",
                    submit_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    error="qsub binary not found",
                )
                return

            _upsert_chunk_row(
                conn,
                chunk_path=cp_str,
                pbs_job_id=job_id,
                status="queued",
                submit_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            logger.info("Submitted %s -> %s", chunk_path.name, job_id)

            # Opportunistic refresh of state for everything we've recorded.
            _refresh_states(conn)
    finally:
        conn.close()


def _refresh_states(conn: sqlite3.Connection) -> None:
    """Update queued/running/completed states for known jobs based on ``qstat``."""

    rows = conn.execute(
        "SELECT chunk_path, pbs_job_id, status FROM sweep_jobs "
        "WHERE pbs_job_id IS NOT NULL AND status IN ('queued', 'running')"
    ).fetchall()
    for row in rows:
        state = _query_job_state(row["pbs_job_id"])
        if state is None:
            # Job left the scheduler — treat as completed (no failure signal
            # we can extract without parsing iqc-parsl exit codes).
            _upsert_chunk_row(
                conn,
                chunk_path=row["chunk_path"],
                pbs_job_id=row["pbs_job_id"],
                status="completed",
                finish_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        elif state == "R":
            if row["status"] != "running":
                _upsert_chunk_row(
                    conn,
                    chunk_path=row["chunk_path"],
                    pbs_job_id=row["pbs_job_id"],
                    status="running",
                )
        elif state in {"Q", "H", "W"}:
            # Still pending — nothing to update.
            continue
        elif state in {"F", "X"}:
            _upsert_chunk_row(
                conn,
                chunk_path=row["chunk_path"],
                pbs_job_id=row["pbs_job_id"],
                status="completed",
                finish_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )


def status(registry_db: Path) -> dict:
    """Return a snapshot of the registry: counts per status + failed-chunk list.

    Result shape:
        {
            "counts": {"queued": 2, "running": 1, "completed": 7, "failed": 1, ...},
            "failed": [{"chunk_path": "...", "pbs_job_id": "...", "error": "..."}, ...],
            "total": 11,
        }
    """

    conn = _open_registry(Path(registry_db))
    try:
        counts: dict[str, int] = {}
        total = 0
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM sweep_jobs GROUP BY status"
        ):
            counts[row["status"]] = row["n"]
            total += row["n"]
        failed = [
            dict(r)
            for r in conn.execute(
                "SELECT chunk_path, pbs_job_id, submit_time, error "
                "FROM sweep_jobs WHERE status = 'failed' ORDER BY id"
            )
        ]
        return {"counts": counts, "failed": failed, "total": total}
    finally:
        conn.close()
