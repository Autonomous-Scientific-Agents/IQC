#!/usr/bin/env python3
"""Standalone wrapper for ``iqc.sweep_orchestrator.chunk_inputs``.

Useful when you want the chunking step alone — without the PBS submission
orchestration that ``iqc-sweep submit`` provides — for example to inspect
the chunk layout before kicking off a sweep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iqc.sweep_orchestrator import chunk_inputs


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an input parquet into PBS-job-sized chunks."
    )
    parser.add_argument("input_parquet", type=Path)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Rows per chunk (overrides the heavy-atom-aware defaults).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the chunk parquets. Defaults to <input>.chunks/.",
    )
    parser.add_argument(
        "--no-heavy-atoms",
        dest="by_heavy_atoms",
        action="store_false",
        help="Disable per-heavy-atom-class sizing; chunk uniformly.",
    )
    parser.set_defaults(by_heavy_atoms=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    chunks = chunk_inputs(
        args.input_parquet,
        chunk_size_hint=args.chunk_size,
        by_heavy_atoms=args.by_heavy_atoms,
        output_dir=args.output_dir,
    )
    for path in chunks:
        print(path)
    return 0


def run_cli() -> int:
    try:
        return main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
