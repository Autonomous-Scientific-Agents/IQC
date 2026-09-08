"""iqc-status: summarize sweep results (ok vs error) across JSONL/SQLite sources.

Used during long Aurora sweeps to answer "how many calcs ran, how many failed,
and what kind of errors are dominating?" without depending on F5's index API.
We read JSONL files line-by-line and SQLite ``calculations.blob_data`` rows
directly so this can run before/independently of the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
from collections import Counter, defaultdict
from typing import Iterable, Iterator


# Recognized task names whose ``{task}_error`` field signals a failed row.
# Kept loose: we also detect any key ending in ``_error`` so that new task
# names added by F1/F10 etc. are handled automatically.
_KNOWN_TASKS = ("scf", "mp2", "ccsd", "t", "energy", "opt", "relax", "nmr")


def _record_error(record: dict) -> str | None:
    """Return the first error value found in a record, else None."""
    # Prefer the task-specific error if ``task`` is present.
    task = record.get("task")
    if isinstance(task, str):
        key = f"{task}_error"
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # Soft in-task failures are recorded in the plain "error" field by the
    # asetools task runners without raising, so no ``{task}_error`` key exists.
    val = record.get("error")
    if isinstance(val, str) and val.strip():
        return val
    if record.get("nonphysical"):
        return "Nonphysical calculation result"
    if record.get("opt_converged") is False:
        return "Optimization did not converge"
    # Fall back to any *_error field (defensive — handles aliasing or sub-task errors).
    for key, value in record.items():
        if key.endswith("_error") and isinstance(value, str) and value.strip():
            return value
    return None


def _iter_jsonl_records(path: pathlib.Path) -> Iterator[dict]:
    """Yield dict records from a JSONL file, skipping malformed lines."""
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _iter_sqlite_records(path: pathlib.Path) -> Iterator[dict]:
    """Yield dict records from an IQC SQLite db's ``calculations`` table."""
    con = sqlite3.connect(str(path))
    try:
        cur = con.execute("SELECT blob_data FROM calculations")
        for (blob,) in cur:
            if not blob:
                continue
            try:
                record = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record
    finally:
        con.close()


def _iter_source_records(source: pathlib.Path) -> Iterator[dict]:
    """Dispatch on suffix to read JSONL or SQLite rows."""
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        # ``.json`` may be a single object or a list-of-objects (rank file).
        if suffix == ".json":
            try:
                with open(source) as h:
                    data = json.load(h)
            except (OSError, json.JSONDecodeError):
                return
            if isinstance(data, dict):
                yield data
            elif isinstance(data, list):
                for r in data:
                    if isinstance(r, dict):
                        yield r
            return
        yield from _iter_jsonl_records(source)
    elif suffix in {".db", ".sqlite", ".sqlite3"}:
        yield from _iter_sqlite_records(source)
    else:
        # Unknown extension — try JSONL as a last resort; many sweep outputs
        # are written without an extension on PBS scratch.
        try:
            yield from _iter_jsonl_records(source)
        except OSError:
            return


def _error_kind(message: str) -> str:
    """Bucket an error message to its first informative line (~80 chars)."""
    first = message.strip().splitlines()[0] if message.strip() else "<empty>"
    return first[:80]


def _passes_filters(record: dict, filters: dict | None) -> bool:
    """Apply caller-supplied filters (task/calculator/status) to a record."""
    if not filters:
        return True
    for field in ("task", "calculator"):
        wanted = filters.get(field)
        if wanted and record.get(field) != wanted:
            return False
    status_filter = filters.get("status")
    if status_filter and status_filter != "all":
        is_err = _record_error(record) is not None
        if status_filter == "ok" and is_err:
            return False
        if status_filter == "error" and not is_err:
            return False
    return True


def summarize(
    sources: Iterable[pathlib.Path],
    *,
    filters: dict | None = None,
) -> dict:
    """Count ok/error rows across JSONL/SQLite sources, broken down by task/calc."""
    counts = {"ok": 0, "error": 0, "total": 0}
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "error": 0})
    by_calc: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "error": 0})
    error_kinds: Counter[str] = Counter()

    for source in sources:
        path = pathlib.Path(source)
        if not path.exists():
            continue
        for record in _iter_source_records(path):
            if not _passes_filters(record, filters):
                continue
            err = _record_error(record)
            status = "error" if err else "ok"
            counts[status] += 1
            counts["total"] += 1
            task = str(record.get("task") or "<unknown>")
            calc = str(record.get("calculator") or "<unknown>")
            by_task[task][status] += 1
            by_calc[calc][status] += 1
            if err:
                error_kinds[_error_kind(err)] += 1

    return {
        "counts": counts,
        "by_task": {k: dict(v) for k, v in by_task.items()},
        "by_calculator": {k: dict(v) for k, v in by_calc.items()},
        "by_error_kind": dict(error_kinds.most_common(10)),
    }


def _format_table(summary: dict) -> str:
    """Render the summary dict as a human-readable plain-text table."""
    lines: list[str] = []
    c = summary["counts"]
    lines.append(f"Total: {c['total']}   OK: {c['ok']}   ERROR: {c['error']}")
    lines.append("")
    if summary["by_task"]:
        lines.append("By task:")
        lines.append(f"  {'task':<20} {'ok':>8} {'error':>8}")
        for task, vals in sorted(summary["by_task"].items()):
            lines.append(f"  {task:<20} {vals['ok']:>8} {vals['error']:>8}")
        lines.append("")
    if summary["by_calculator"]:
        lines.append("By calculator:")
        lines.append(f"  {'calculator':<20} {'ok':>8} {'error':>8}")
        for calc, vals in sorted(summary["by_calculator"].items()):
            lines.append(f"  {calc:<20} {vals['ok']:>8} {vals['error']:>8}")
        lines.append("")
    if summary["by_error_kind"]:
        lines.append("Top error kinds:")
        for kind, n in summary["by_error_kind"].items():
            lines.append(f"  {n:>6}  {kind}")
    return "\n".join(lines)


def _expand_jsonl_globs(patterns: list[str]) -> list[pathlib.Path]:
    """Expand each glob pattern; literal paths are accepted as-is."""
    out: list[pathlib.Path] = []
    for pattern in patterns:
        p = pathlib.Path(pattern)
        # If the literal path exists, take it; otherwise treat as a glob.
        if p.exists():
            out.append(p)
            continue
        # Glob relative to cwd, accepting absolute patterns too.
        if p.is_absolute():
            root = pathlib.Path(p.anchor)
            rel = str(p.relative_to(root))
            matches = list(root.glob(rel))
        else:
            matches = list(pathlib.Path().glob(pattern))
        out.extend(matches)
    return out


def _build_parser() -> argparse.ArgumentParser:
    """Build the iqc-status argparse parser."""
    p = argparse.ArgumentParser(
        prog="iqc-status",
        description="Summarize ok/error counts from IQC sweep JSONL/SQLite outputs.",
    )
    p.add_argument(
        "--db",
        action="append",
        default=[],
        help="Path to an IQC SQLite database (.db). Repeatable.",
    )
    p.add_argument(
        "--jsonl",
        action="append",
        default=[],
        help="Path or glob to a JSONL result file. Repeatable.",
    )
    p.add_argument(
        "--status",
        choices=("ok", "error", "all"),
        default="all",
        help="Only count rows matching this status (default: all).",
    )
    p.add_argument("--task", default=None, help="Restrict to this task name.")
    p.add_argument(
        "--calculator", default=None, help="Restrict to this calculator name."
    )
    fmt = p.add_mutually_exclusive_group()
    fmt.add_argument(
        "--json",
        action="store_const",
        dest="format",
        const="json",
        help="Emit JSON summary.",
    )
    fmt.add_argument(
        "--table",
        action="store_const",
        dest="format",
        const="table",
        help="Emit a plain-text table (default).",
    )
    p.set_defaults(format="table")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``iqc-status``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    sources: list[pathlib.Path] = [pathlib.Path(d) for d in args.db]
    sources.extend(_expand_jsonl_globs(args.jsonl))
    if not sources:
        parser.error("at least one --db or --jsonl source is required")
    filters = {
        "status": args.status,
        "task": args.task,
        "calculator": args.calculator,
    }
    summary = summarize(sources, filters=filters)
    if args.format == "json":
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_format_table(summary) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
