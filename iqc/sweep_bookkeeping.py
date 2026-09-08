"""Scalable, multi-user-safe bookkeeping for large IQC sweeps.

This is the query + claiming layer of the sweep-bookkeeping redesign. It
replaces the old text ledgers (``done.txt`` / ``claimed.txt`` /
``skip_index.jsonl``) and the O(all-files) ``rebuild`` harvest with:

* **State derived on read** — the *done* / *remaining* sets are computed with a
  single DuckDB query over the immutable per-job parquet/JSONL glob. Read-only,
  so any number of users/readers are safe by construction, and one place
  defines "done" (a row with non-null energy) → no semantic drift between
  tools.
* **Atomic chunk claiming** — a submitter claims a whole chunk by
  ``os.rename('.../todo/c.parquet', '.../claimed/<user>/c.parquet')``. POSIX /
  Lustre ``rename`` is atomic, so exactly one user wins a given chunk and the
  losers move on. A completed chunk moves to ``done/``; a failed one moves back
  to ``todo/`` (auto re-eligible — no manual reconcile).

Identity note (the reason there is no regex in the common path): results carry
a stored ``unique_name_base`` column (written by ``iqc.main._process_one_row``)
that equals the input identity without the run-id suffix, so the done/remaining
join is an exact column equality. Legacy result files predate that column; for
those rows we fall back to stripping the suffix off ``unique_name`` with the
documented pattern below. New rows never need the fallback.

DuckDB is an optional dependency (``pip install 'iqc[bookkeeping]'``); it is
imported lazily so importing this module never requires it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

# Run-id suffix appended by ``_process_one_row``:
#   unique_name = f"{base}_{xyz_index}_{worker_id}_{YYYYMMDD}_{HHMMSS}"
# Used ONLY as a fallback for legacy rows that lack ``unique_name_base``.
_RUNID_SUFFIX_RE = r"_[0-9]+_[0-9]+_[0-9]{8}_[0-9]{6}$"

DEFAULT_ENERGY_COLUMN = "total_energy_eV"


def _require_duckdb():
    """Import duckdb lazily with an actionable error if it is missing."""
    try:
        import duckdb  # noqa: WPS433 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The sweep bookkeeping query layer needs DuckDB. Install it with "
            "`pip install duckdb` or `pip install 'iqc[bookkeeping]'`."
        ) from exc
    return duckdb


def base_from_unique_name(unique_name: str) -> str:
    """Strip the run-id suffix off a full ``unique_name`` (legacy fallback).

    Prefer the stored ``unique_name_base`` column; this is only for rows that
    predate it. Idempotent: a name without a recognizable suffix is returned
    unchanged.
    """
    import re

    return re.sub(_RUNID_SUFFIX_RE, "", unique_name)


def _parquet_columns(duckdb, glob: str) -> set[str]:
    """Return the union of column names across the parquet/JSONL glob."""
    rows = duckdb.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{glob}', union_by_name=true)"
    ).fetchall()
    return {r[0] for r in rows}


def _base_expr(columns: Iterable[str]) -> str:
    """SQL expression yielding the base identity for a result row.

    Uses the stored ``unique_name_base`` when the column exists (COALESCEd with
    the regex fallback for rows within a mixed file that happen to be null),
    otherwise strips the suffix off ``unique_name``.
    """
    cols = set(columns)
    strip = f"regexp_replace(unique_name, '{_RUNID_SUFFIX_RE}', '')"
    if "unique_name_base" in cols:
        return f"COALESCE(unique_name_base, {strip})"
    if "unique_name" in cols:
        return strip
    raise ValueError(
        "result files have neither 'unique_name_base' nor 'unique_name'; "
        "cannot derive an identity to compute the done-set."
    )


def _connect(duckdb):
    """A fresh in-memory read-only connection (many readers are safe)."""
    return duckdb.connect(database=":memory:")


def done_uids(
    results_glob: str,
    *,
    energy_column: str = DEFAULT_ENERGY_COLUMN,
) -> set[str]:
    """Return the set of base identities that are *done*.

    A base identity is done iff at least one of its result rows has a non-null
    energy. This is the single, authoritative definition of "done" shared by
    every tool (fixes the historical drift where a null-energy row counted as
    done in one index but not in another).
    """
    duckdb = _require_duckdb()
    con = _connect(duckdb)
    cols = _parquet_columns(con, results_glob)
    if energy_column not in cols:
        raise ValueError(
            f"energy column {energy_column!r} not found in {results_glob}; "
            f"available columns include: {sorted(cols)[:12]}..."
        )
    base = _base_expr(cols)
    rows = con.sql(
        f"""
        SELECT DISTINCT {base} AS uid
        FROM read_parquet('{results_glob}', union_by_name=true)
        WHERE {energy_column} IS NOT NULL
        """
    ).fetchall()
    return {r[0] for r in rows}


def remaining(
    input_parquet: str,
    results_glob: str,
    *,
    uid_column: str = "unique_name",
    energy_column: str = DEFAULT_ENERGY_COLUMN,
) -> list[str]:
    """Return input identities with no done result yet (an ANTI JOIN).

    ``input_parquet`` supplies the authoritative work list; ``uid_column`` is
    its stable identity column (``unique_name`` for the sweep, e.g.
    ``C10H22_conf0000``). Result rows are keyed by their base identity so the
    join is exact — no suffix parsing on the common path.
    """
    duckdb = _require_duckdb()
    con = _connect(duckdb)
    rcols = _parquet_columns(con, results_glob)
    base = _base_expr(rcols)
    if energy_column not in rcols:
        raise ValueError(
            f"energy column {energy_column!r} not found in {results_glob}."
        )
    rows = con.sql(
        f"""
        WITH done AS (
            SELECT DISTINCT {base} AS uid
            FROM read_parquet('{results_glob}', union_by_name=true)
            WHERE {energy_column} IS NOT NULL
        )
        SELECT i.uid
        FROM (
            SELECT DISTINCT {uid_column} AS uid
            FROM read_parquet('{input_parquet}', union_by_name=true)
        ) i
        ANTI JOIN done d ON i.uid = d.uid
        ORDER BY i.uid
        """
    ).fetchall()
    return [r[0] for r in rows]


def summary(
    results_glob: str,
    *,
    energy_column: str = DEFAULT_ENERGY_COLUMN,
) -> dict:
    """Return {total_rows, distinct_uids, done_uids, error_rows}."""
    duckdb = _require_duckdb()
    con = _connect(duckdb)
    cols = _parquet_columns(con, results_glob)
    base = _base_expr(cols)
    has_energy = energy_column in cols
    energy_ok = f"{energy_column} IS NOT NULL" if has_energy else "FALSE"
    row = con.sql(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT {base}) AS distinct_uids,
            COUNT(DISTINCT CASE WHEN {energy_ok} THEN {base} END) AS done_uids
        FROM read_parquet('{results_glob}', union_by_name=true)
        """
    ).fetchone()
    return {
        "total_rows": row[0],
        "distinct_uids": row[1],
        "done_uids": row[2],
        "not_done_uids": row[1] - row[2],
    }


# --------------------------------------------------------------------------- #
# Atomic chunk claiming
# --------------------------------------------------------------------------- #


def list_todo(todo_dir: os.PathLike | str, pattern: str = "*.parquet") -> list[Path]:
    """Return unclaimed chunk files, sorted (stable pick order)."""
    return sorted(Path(todo_dir).glob(pattern))


def claim_chunk(
    todo_dir: os.PathLike | str,
    claimed_root: os.PathLike | str,
    user: str,
    *,
    pattern: str = "*.parquet",
) -> Optional[Path]:
    """Atomically claim one chunk; return its new path, or None if none free.

    Iterates candidate chunks in ``todo_dir`` and tries to ``os.rename`` each
    into ``claimed_root/<user>/``. ``rename`` is atomic on POSIX/Lustre, so if
    two users race for the same chunk exactly one ``rename`` succeeds; the loser
    gets ``FileNotFoundError`` and simply tries the next candidate. Returns None
    only when every candidate was taken by someone else (or the dir is empty).
    """
    user_dir = Path(claimed_root) / user
    user_dir.mkdir(parents=True, exist_ok=True)
    for chunk in list_todo(todo_dir, pattern):
        dest = user_dir / chunk.name
        try:
            os.rename(chunk, dest)
        except (FileNotFoundError, OSError):
            # Lost the race (someone renamed it first) — try the next one.
            continue
        return dest
    return None


def _move_chunk(chunk: os.PathLike | str, dest_dir: os.PathLike | str) -> Path:
    chunk = Path(chunk)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / chunk.name
    os.rename(chunk, dest)
    return dest


def complete_chunk(chunk: os.PathLike | str, done_dir: os.PathLike | str) -> Path:
    """Move a finished chunk to ``done_dir`` (atomic rename)."""
    return _move_chunk(chunk, done_dir)


def fail_chunk(chunk: os.PathLike | str, todo_dir: os.PathLike | str) -> Path:
    """Return a failed chunk to ``todo_dir`` so it is auto re-eligible."""
    return _move_chunk(chunk, todo_dir)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iqc-sweep-book",
        description="Query done/remaining and claim chunks for a sweep.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_energy(sp):
        sp.add_argument(
            "--energy-column",
            default=DEFAULT_ENERGY_COLUMN,
            help=f"Column whose non-null value means 'done' (default: {DEFAULT_ENERGY_COLUMN}).",
        )

    d = sub.add_parser("done", help="Print done base identities (or --count).")
    d.add_argument("--results", required=True, help="Glob of per-job result files.")
    d.add_argument("--count", action="store_true", help="Print only the count.")
    add_energy(d)

    r = sub.add_parser("remaining", help="Print inputs with no done result yet.")
    r.add_argument("--input", required=True, help="Input parquet (the work list).")
    r.add_argument("--results", required=True, help="Glob of per-job result files.")
    r.add_argument("--uid-column", default="unique_name", help="Identity column of --input.")
    r.add_argument("--count", action="store_true", help="Print only the count.")
    add_energy(r)

    s = sub.add_parser("summary", help="Print row/uid/done counts.")
    s.add_argument("--results", required=True, help="Glob of per-job result files.")
    add_energy(s)

    c = sub.add_parser("claim", help="Atomically claim one chunk from todo/.")
    c.add_argument("--todo", required=True)
    c.add_argument("--claimed", required=True, help="Root under which <user>/ is created.")
    c.add_argument("--user", required=True)

    cp = sub.add_parser("complete", help="Move a chunk to done/.")
    cp.add_argument("--chunk", required=True)
    cp.add_argument("--done", required=True)

    f = sub.add_parser("fail", help="Return a chunk to todo/ (auto re-eligible).")
    f.add_argument("--chunk", required=True)
    f.add_argument("--todo", required=True)

    lt = sub.add_parser("list-todo", help="List unclaimed chunks.")
    lt.add_argument("--todo", required=True)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "done":
        uids = done_uids(args.results, energy_column=args.energy_column)
        if args.count:
            print(len(uids))
        else:
            for u in sorted(uids):
                print(u)
        return 0

    if args.cmd == "remaining":
        rem = remaining(
            args.input,
            args.results,
            uid_column=args.uid_column,
            energy_column=args.energy_column,
        )
        if args.count:
            print(len(rem))
        else:
            for u in rem:
                print(u)
        return 0

    if args.cmd == "summary":
        summ = summary(args.results, energy_column=args.energy_column)
        for k, v in summ.items():
            print(f"{k}: {v}")
        return 0

    if args.cmd == "claim":
        claimed = claim_chunk(args.todo, args.claimed, args.user)
        if claimed is None:
            print("no chunk available", file=sys.stderr)
            return 1
        print(claimed)
        return 0

    if args.cmd == "complete":
        print(complete_chunk(args.chunk, args.done))
        return 0

    if args.cmd == "fail":
        print(fail_chunk(args.chunk, args.todo))
        return 0

    if args.cmd == "list-todo":
        for c in list_todo(args.todo):
            print(c)
        return 0

    return 2  # pragma: no cover - argparse enforces a subcommand


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
