"""Scalable, multi-user-safe bookkeeping for large IQC sweeps.

This is the query + claiming layer of the sweep-bookkeeping redesign. It
replaces the old text ledgers (``done.txt`` / ``claimed.txt`` /
``skip_index.jsonl``) and the O(all-files) ``rebuild`` harvest with:

* **State derived on read** — the *done* / *remaining* sets are computed with a
  single DuckDB query over the immutable per-job parquet/JSONL glob. Read-only,
  so any number of users/readers are safe by construction, and one place
  defines "done" — a *successful* row (finite energy, no ``error``/``*_error``,
  not ``nonphysical``, ``opt_converged`` not false), matching
  ``status_query`` → no semantic drift between tools.
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

# IQC energy columns end in ``_eV`` (total_energy_eV, energy_eV, scf_energy_eV,
# opt_total_energy_eV, initial_*_eV, ...). These are candidate values for
# missing-column validation; recorded failures must be excluded first.
_ENERGY_COLUMN_SUFFIX = "_eV"


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


def _connect(duckdb):
    """A fresh in-memory read-only connection (many readers are safe)."""
    return duckdb.connect(database=":memory:")


def _quote(ident: str) -> str:
    """Double-quote a SQL identifier, escaping embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _sql_str_list(paths: Iterable[str]) -> str:
    """Render paths as a SQL list literal: ['a', 'b'] with quotes escaped."""
    return "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"


def _glob_files(con, glob: str) -> list[str]:
    """Return the files matching ``glob`` (empty list when none exist).

    DuckDB's ``glob`` table function returns zero rows for a pattern that
    matches nothing — this is how we tell "no results yet" (a normal fresh-sweep
    state) apart from "a file exists but is unreadable" (a real error that the
    subsequent read surfaces).
    """
    return [r[0] for r in con.execute("SELECT file FROM glob(?)", [glob]).fetchall()]


def _scan_sql(files: list[str]) -> str:
    """A ``FROM``-able subquery reading every file, keyed by column name.

    Parquet files are read with ``read_parquet``; everything else (``.jsonl``
    and extensionless sweep output) with ``read_json`` in newline-delimited mode
    with ``ignore_errors`` so a walltime-truncated last line is dropped while the
    valid rows before it are kept. Mixed globs are UNIONed by name.
    """
    parquet = [f for f in files if f.lower().endswith((".parquet", ".pq"))]
    jsonish = [f for f in files if f not in parquet]
    parts: list[str] = []
    if parquet:
        parts.append(
            f"SELECT * FROM read_parquet({_sql_str_list(parquet)}, union_by_name=true)"
        )
    if jsonish:
        parts.append(
            "SELECT * FROM read_json("
            f"{_sql_str_list(jsonish)}, union_by_name=true, "
            "format='newline_delimited', ignore_errors=true)"
        )
    return " UNION ALL BY NAME ".join(f"({p})" for p in parts)


def _describe(con, scan_sql: str) -> dict[str, str]:
    """Return {column_name: upper-case column_type} for a scan subquery."""
    rows = con.execute(f"DESCRIBE SELECT * FROM ({scan_sql})").fetchall()
    return {r[0]: str(r[1]).upper() for r in rows}


def _base_expr(columns: Iterable[str]) -> str:
    """SQL expression yielding the base identity for a result row.

    ``unique_name_base`` is the canonical key. Only when *both* it and the
    legacy ``unique_name`` exist do we ``COALESCE`` (so a mixed file with some
    null bases still resolves); referencing ``unique_name`` when the column is
    absent would raise a binder error even if every base is non-null.
    """
    cols = set(columns)
    strip = f"regexp_replace(unique_name, '{_RUNID_SUFFIX_RE}', '')"
    has_base = "unique_name_base" in cols
    has_name = "unique_name" in cols
    if has_base and has_name:
        return f"COALESCE(unique_name_base, {strip})"
    if has_base:
        return "unique_name_base"
    if has_name:
        return strip
    raise ValueError(
        "result files have neither 'unique_name_base' nor 'unique_name'; "
        "cannot derive an identity to compute the done-set."
    )


def _no_failure_predicate(schema: dict[str, str]) -> str:
    """Rows without recorded failures, independent of available energy columns."""
    parts = []
    if "nonphysical" in schema:
        parts.append("(nonphysical IS NULL OR nonphysical = FALSE)")
    if "opt_converged" in schema:
        parts.append("(opt_converged IS NULL OR opt_converged = TRUE)")
    for c in sorted(schema):
        if c == "error" or c.endswith("_error"):
            q = _quote(c)
            parts.append(f"({q} IS NULL OR {q} = '')")
    return " AND ".join(parts) or "TRUE"


def _success_predicate(schema: dict[str, str], energy_column: str) -> str:
    """SQL predicate for a *successful* result row.

    A non-null energy is necessary but not sufficient: IQC keeps a final energy
    on an exhausted optimization and on later-stage failures. Mirrors
    ``status_query._record_error`` so done/remaining/summary agree with the
    status tooling: finite energy AND no ``error``/``*_error`` AND not
    ``nonphysical`` AND ``opt_converged`` not false.
    """
    cols = set(schema)
    if energy_column not in cols:
        raise ValueError(
            f"energy column {energy_column!r} not found; available columns "
            f"include: {sorted(cols)[:12]}..."
        )
    e = _quote(energy_column)
    parts = [f"{e} IS NOT NULL", _no_failure_predicate(schema)]
    if any(t in schema[energy_column] for t in ("DOUBLE", "FLOAT", "REAL", "DECIMAL")):
        parts.append(f"NOT isnan({e})")
        parts.append(f"NOT isinf({e})")
    return " AND ".join(parts)


def _classify_source(
    con, scan: str, schema: dict[str, str], energy_column: str
) -> tuple[bool, bool]:
    """Return ``(has_identity, has_energy)`` for a result schema.

    A source made up only of failed/interrupted rows legitimately lacks the
    energy and/or identity columns — a backend that raises before writing an
    energy leaves ``unique_name_base`` + ``{task}_error`` and no energy; an EL
    ``_synthesize_failure_row`` has neither identity column; an empty JSONL
    (opened before the first completion) has neither. These contribute *no done
    UIDs* (so their inputs stay eligible for retry) rather than erroring.

    If the requested column is absent, only rows without recorded failures
    and with an actual energy value can establish a missing-column error.
    Failed later stages may retain intermediate energies; those must not block
    retries. Empty/all-null rows left by an interrupted append also contribute
    no evidence of a misspelled column. The ordinary success path needs no
    extra scan. Unreadable/corrupt files raise earlier, in :func:`_describe`.
    """
    has_identity = "unique_name_base" in schema or "unique_name" in schema
    has_energy = energy_column in schema
    present = sorted(c for c in schema if c.endswith(_ENERGY_COLUMN_SUFFIX))
    if not has_energy and present:
        any_energy = " OR ".join(f"{_quote(c)} IS NOT NULL" for c in present)
        candidate = _no_failure_predicate(schema)
        has_candidate = con.execute(
            f"SELECT EXISTS (SELECT 1 FROM ({scan}) "
            f"WHERE ({candidate}) AND ({any_energy}))"
        ).fetchone()[0]
        if has_candidate:
            raise ValueError(
                f"energy column {energy_column!r} not found; available energy "
                f"columns: {present}"
            )
    return has_identity, has_energy


def _empty_summary() -> dict:
    return {"total_rows": 0, "distinct_uids": 0, "done_uids": 0, "not_done_uids": 0}


def done_uids(
    results_glob: str,
    *,
    energy_column: str = DEFAULT_ENERGY_COLUMN,
) -> set[str]:
    """Return the set of base identities that have a *successful* result.

    One authoritative definition of "done" shared by every tool (fixes the
    historical drift where a null-energy row counted as done in one index but
    not another). A result set with no successful rows — an empty glob (fresh
    sweep) or only failed/interrupted output — yields an empty set rather than
    an error.
    """
    duckdb = _require_duckdb()
    con = _connect(duckdb)
    files = _glob_files(con, results_glob)
    if not files:
        return set()
    scan = _scan_sql(files)
    schema = _describe(con, scan)
    has_identity, has_energy = _classify_source(con, scan, schema, energy_column)
    if not (has_identity and has_energy):
        return set()
    base = _base_expr(schema)
    success = _success_predicate(schema, energy_column)
    rows = con.execute(
        f"SELECT DISTINCT {base} AS uid FROM ({scan}) WHERE {success}"
    ).fetchall()
    return {r[0] for r in rows}


def remaining(
    input_parquet: str,
    results_glob: str,
    *,
    uid_column: str = "unique_name",
    energy_column: str = DEFAULT_ENERGY_COLUMN,
) -> list[str]:
    """Return input identities with no successful result yet (an ANTI JOIN).

    ``input_parquet`` supplies the authoritative work list; ``uid_column`` is
    its stable identity column (``unique_name`` for the sweep). Result rows are
    keyed by their stored base identity so the join is exact — no suffix parsing
    on the common path. When the result set has no successful rows — no file yet
    (fresh sweep) or only failed/interrupted output — every input UID is
    returned so nothing is dropped from the retry list.
    """
    duckdb = _require_duckdb()
    con = _connect(duckdb)
    uid = _quote(uid_column)
    input_lit = input_parquet.replace("'", "''")
    input_scan = (
        f"SELECT DISTINCT {uid} AS uid "
        f"FROM read_parquet('{input_lit}', union_by_name=true)"
    )

    def _all_inputs() -> list[str]:
        rows = con.execute(f"SELECT uid FROM ({input_scan}) ORDER BY uid").fetchall()
        return [r[0] for r in rows]

    files = _glob_files(con, results_glob)
    if not files:
        return _all_inputs()
    scan = _scan_sql(files)
    schema = _describe(con, scan)
    has_identity, has_energy = _classify_source(con, scan, schema, energy_column)
    if not (has_identity and has_energy):
        return _all_inputs()
    base = _base_expr(schema)
    success = _success_predicate(schema, energy_column)
    rows = con.execute(
        f"""
        WITH done AS (
            SELECT DISTINCT {base} AS uid FROM ({scan}) WHERE {success}
        )
        SELECT i.uid FROM ({input_scan}) i
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
    """Return {total_rows, distinct_uids, done_uids, not_done_uids}.

    Zero counts for an empty result glob (fresh sweep). A failure-only source
    still reports its row/identity counts (where an identity column exists) with
    ``done_uids`` = 0, rather than erroring.
    """
    duckdb = _require_duckdb()
    con = _connect(duckdb)
    files = _glob_files(con, results_glob)
    if not files:
        return _empty_summary()
    scan = _scan_sql(files)
    schema = _describe(con, scan)
    has_identity, has_energy = _classify_source(con, scan, schema, energy_column)
    if has_identity:
        base = _base_expr(schema)
        distinct_expr = f"COUNT(DISTINCT {base})"
        done_expr = (
            f"COUNT(DISTINCT CASE WHEN {_success_predicate(schema, energy_column)} "
            f"THEN {base} END)"
            if has_energy
            else "0"
        )
    else:
        distinct_expr = "0"
        done_expr = "0"
    row = con.execute(
        f"SELECT COUNT(*), {distinct_expr}, {done_expr} FROM ({scan})"
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
