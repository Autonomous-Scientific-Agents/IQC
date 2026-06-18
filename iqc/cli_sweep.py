"""``iqc-sweep`` console entry point.

Thin argparse front-end over :mod:`iqc.sweep_orchestrator`. Three subcommands:

- ``chunk``   — slice an input parquet into PBS-job-sized pieces.
- ``submit``  — feed chunks into PBS, throttled on the user's queue cap.
- ``status``  — read the registry SQLite and print counts + failed chunks.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from iqc.sweep_orchestrator import chunk_inputs, status, submit_sweep


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iqc-sweep",
        description=(
            "Chunk a large input parquet and submit iqc-parsl PBS jobs while "
            "respecting Aurora's per-user queued-job cap."
        ),
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chunk = sub.add_parser("chunk", help="Split an input parquet into chunk parquets.")
    chunk.add_argument("input_parquet", type=Path, help="Path to the input parquet.")
    chunk.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Rows per chunk. When omitted and --by-heavy-atoms is on, a per-class "
            "default is used (small molecules many rows, large molecules few)."
        ),
    )
    chunk.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write chunks into. Defaults to <input>.chunks/.",
    )
    chunk.add_argument(
        "--no-heavy-atoms",
        dest="by_heavy_atoms",
        action="store_false",
        help="Disable heavy-atom-aware chunk sizing; chunk uniformly.",
    )
    chunk.set_defaults(by_heavy_atoms=True)

    submit = sub.add_parser("submit", help="Submit iqc-parsl jobs for each chunk parquet.")
    submit.add_argument(
        "chunk_dir",
        type=Path,
        help="Directory of chunk parquets (e.g. the output of `iqc-sweep chunk`).",
    )
    submit.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="SQLite registry path. Created if missing.",
    )
    submit.add_argument(
        "--max-queued",
        type=int,
        default=4,
        help="Maximum number of user-owned PBS jobs to keep in queue at once.",
    )
    submit.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between qstat polls when the queue cap is hit.",
    )
    submit.add_argument(
        "--queue",
        default="prod",
        help="Target PBS queue (debug, prod, ...).",
    )
    submit.add_argument(
        "--walltime",
        default="12:00:00",
        help="PBS walltime per job, H:MM:SS.",
    )
    submit.add_argument("--account", default="IQC", help="PBS charging account.")
    submit.add_argument("--nodes", type=int, default=1, help="Aurora nodes per PBS job.")
    submit.add_argument(
        "--filesystems",
        default="home:flare",
        help="PBS -l filesystems= value.",
    )
    submit.add_argument(
        "--venv-activate",
        default=None,
        help="Absolute path to the venv's bin/activate.",
    )
    submit.add_argument(
        "--script-dir",
        type=Path,
        default=None,
        help="Where to write generated PBS scripts (default ./pbs_scripts).",
    )
    submit.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="PBS stdout/stderr destination (default ./pbs_logs).",
    )
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Record 'would_submit' in the registry without invoking qsub.",
    )
    submit.add_argument(
        "iqc_parsl_args",
        nargs=argparse.REMAINDER,
        help=(
            "Everything after '--' is forwarded verbatim to iqc-parsl. "
            "Example: -- --calculator exachem --task ccsdt"
        ),
    )

    st = sub.add_parser("status", help="Print registry counts + failed chunks.")
    st.add_argument("--registry", type=Path, required=True, help="SQLite registry path.")
    st.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit JSON instead of human-friendly text.",
    )

    return parser


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="iqc-sweep %(levelname)s %(asctime)s %(message)s",
    )


def _run_chunk(ns: argparse.Namespace) -> int:
    chunks = chunk_inputs(
        ns.input_parquet,
        chunk_size_hint=ns.chunk_size,
        by_heavy_atoms=ns.by_heavy_atoms,
        output_dir=ns.output_dir,
    )
    for path in chunks:
        print(path)
    return 0


def _run_submit(ns: argparse.Namespace) -> int:
    if not ns.chunk_dir.is_dir():
        print(f"Not a directory: {ns.chunk_dir}", file=sys.stderr)
        return 1
    chunk_paths = sorted(ns.chunk_dir.glob("*.parquet"))
    if not chunk_paths:
        print(f"No parquet chunks found in {ns.chunk_dir}", file=sys.stderr)
        return 1

    # Drop the leading "--" that argparse REMAINDER preserves so the
    # forwarded args look natural to iqc-parsl.
    forwarded = list(ns.iqc_parsl_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    submit_sweep(
        chunk_paths,
        registry_db=ns.registry,
        max_queued=ns.max_queued,
        poll_interval_s=ns.poll_interval,
        iqc_parsl_args=forwarded,
        dry_run=ns.dry_run,
        queue=ns.queue,
        walltime=ns.walltime,
        account=ns.account,
        nodes=ns.nodes,
        filesystems=ns.filesystems,
        venv_activate=ns.venv_activate,
        script_dir=ns.script_dir,
        log_dir=ns.log_dir,
    )
    return 0


def _run_status(ns: argparse.Namespace) -> int:
    snap = status(ns.registry)
    if ns.as_json:
        print(json.dumps(snap, indent=2, sort_keys=True))
        return 0
    print(f"Total chunks recorded: {snap['total']}")
    if snap["counts"]:
        print("Counts by status:")
        for key in sorted(snap["counts"]):
            print(f"  {key:14s} {snap['counts'][key]}")
    if snap["failed"]:
        print(f"\nFailed chunks ({len(snap['failed'])}):")
        for row in snap["failed"]:
            err = (row.get("error") or "").splitlines()[0][:120]
            print(f"  - {row['chunk_path']}  job={row.get('pbs_job_id')}  err={err}")
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    _setup_logging(ns.loglevel)
    if ns.command == "chunk":
        return _run_chunk(ns)
    if ns.command == "submit":
        return _run_submit(ns)
    if ns.command == "status":
        return _run_status(ns)
    parser.error(f"Unknown command: {ns.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
