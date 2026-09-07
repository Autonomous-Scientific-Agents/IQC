"""Walltime estimator for HiFiThermKin sweep chunks.

Scans past run JSONL files under a runs root and extracts per-molecule
ExaChem timings as a function of ``(num_heavy_atoms, nodes_per_mol)``,
caches the resulting stats, and exposes ``estimate_walltime`` for a
prospective chunk + (total_nodes, nodes_per_mol) submission.

Per-molecule wall time we attribute to a run is
``scf_time_s + ccsd_time_s + t_time_s`` (the three timing fields IQC
writes per record). A small fixed startup overhead is added on top.

Caches stats to ``<runs_root>/../results/timing_stats.parquet`` (read by
``iqc-sweep estimate`` and ``submit_chunk.sh``). Rebuild whenever new
runs land; the JSONL scan is the slow step.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Default locations for the HiFiThermKin shared sweep.
DEFAULT_SWEEP_ROOT = Path("/lus/flare/projects/HiFiThermKin/sweep")
DEFAULT_RUNS_ROOT = DEFAULT_SWEEP_ROOT / "runs"
DEFAULT_CACHE_PATH = DEFAULT_SWEEP_ROOT / "results" / "timing_stats.parquet"

# Fields we sum to get per-molecule wall time (seconds).
_TIME_FIELDS = ("scf_time_s", "ccsd_time_s", "t_time_s")

# nodes_per_mol policy keyed by heavy-atom count. Each entry is
# ``(max_h, npm)``; we pick the first entry whose ``max_h`` is ≥ the
# chunk's largest heavy-atom value. The last entry is the catch-all.
# Calibrated from real measurements; ``estimate`` will warn if a chunk
# has no historical timings to corroborate.
NPM_POLICY: tuple[tuple[int, int], ...] = (
    (5, 1),
    (6, 2),
    (7, 4),
    (8, 8),
    (9, 16),
    (10**6, 32),
)

# Queue routing policy keyed by heavy-atom count. h≤5 chunks have a known
# CCSD straggler problem (some conformers — esp. C5H12 — hit a slow
# CCSD-initialization path on Aurora ~50% of the time, needing >40 min
# even at npm=1). The 1 h cap of debug-scaling clips them; capacity's
# 168 h cap absorbs the tail. Larger h-classes have predictable per-mol
# wall, so the routing-via-prod default works fine.
QUEUE_POLICY: tuple[tuple[int, str], ...] = (
    (5, "capacity"),
    (10**6, "prod"),
)

# Per-job startup overhead (seconds): Parsl + MPI + ExaChem cold-start,
# plus the trailing artifact-archive step. Empirical floor; raise if a
# given chunk has lots of mols and the launch overhead dominates.
DEFAULT_STARTUP_OVERHEAD_S = 300.0

# Multiplicative safety factor on top of the predicted wallclock. 1.3
# absorbs per-mol variance + the long-tail straggler that delays a batch.
DEFAULT_SAFETY_FACTOR = 1.3

# PBS queue walltime caps (seconds) on Aurora. Used to clamp the
# estimate so we don't generate an HH:MM:SS that the scheduler refuses.
# Falls back to "no clamp" for unknown queue names.
QUEUE_WALLTIME_CAPS_S = {
    "debug": 60 * 60,
    "debug-scaling": 60 * 60,
    "capacity": 168 * 60 * 60,
    "small": 12 * 60 * 60,
    "medium": 18 * 60 * 60,
    "large": 24 * 60 * 60,
    "backfill-small": 12 * 60 * 60,
    "backfill-medium": 18 * 60 * 60,
    "backfill-large": 24 * 60 * 60,
    # ``prod`` is a routing queue — actual cap depends on routed-to queue;
    # use small's 12 h as a conservative ceiling for the typical case.
    "prod": 12 * 60 * 60,
    "prod-large": 24 * 60 * 60,
}

# Run-dir name encodes nodes_per_mol as ``_<N>n_<M>npm_``; parse with this.
_RUNDIR_NPM_RE = re.compile(r"_(\d+)n_(\d+)npm_")


@dataclass(frozen=True)
class TimingStat:
    """Aggregated per-molecule wall time for one (heavy_atoms, npm) cell."""

    num_heavy_atoms: int
    nodes_per_mol: int
    samples: int
    median_s: float
    p90_s: float
    mean_s: float


# ---------------------------------------------------------------------------
# JSONL scan + cache
# ---------------------------------------------------------------------------


def _parse_npm_from_rundir(name: str) -> Optional[int]:
    """Extract nodes_per_mol from a run-dir name (returns None if absent)."""
    m = _RUNDIR_NPM_RE.search(name)
    return int(m.group(2)) if m else None


def _iter_record_timings(
    runs_root: Path,
) -> Iterable[tuple[int, int, float]]:
    """Yield ``(num_heavy_atoms, nodes_per_mol, total_time_s)`` per JSONL record.

    Records without all three timing fields, without a heavy-atoms count,
    or whose parent rundir name has no ``_<N>npm_`` token are skipped.
    """
    for rundir in runs_root.iterdir():
        if not rundir.is_dir():
            continue
        npm = _parse_npm_from_rundir(rundir.name)
        if npm is None:
            continue
        for jsonl_path in rundir.glob("iqc_single_results_*.jsonl"):
            try:
                with jsonl_path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        h = rec.get("num_heavy_atoms")
                        if h is None:
                            # Fall back to counting non-H atoms in the formula.
                            h = _heavy_from_formula(rec.get("formula", ""))
                        if h is None:
                            continue
                        try:
                            total = sum(float(rec[k]) for k in _TIME_FIELDS)
                        except (KeyError, TypeError, ValueError):
                            continue
                        if total <= 0:
                            continue
                        yield int(h), npm, total
            except OSError as e:
                logger.warning("could not read %s: %s", jsonl_path, e)


# Match every element symbol; hydrogen is skipped in the loop below.
# ([A-GI-Z] excluded H as a first letter, which also dropped He/Hf/Hg/Ho/Hs
# and shifted those molecules into the wrong timing bucket.)
_HEAVY_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _heavy_from_formula(formula: str) -> Optional[int]:
    """Best-effort heavy-atom count from a Hill formula string."""
    if not formula:
        return None
    total = 0
    for sym, n in _HEAVY_FORMULA_RE.findall(formula):
        if not sym or sym == "H":
            continue
        total += int(n) if n else 1
    return total or None


def build_timing_cache(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> Path:
    """Rebuild the (heavy_atoms, npm) timing-stats cache parquet."""
    from collections import defaultdict
    import statistics

    bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    n_records = 0
    for h, npm, sec in _iter_record_timings(runs_root):
        bucket[(h, npm)].append(sec)
        n_records += 1

    stats: list[TimingStat] = []
    for (h, npm), times in sorted(bucket.items()):
        times.sort()
        n = len(times)
        median = statistics.median(times)
        # p90 via nearest-rank; for small n falls back to max.
        p90_idx = max(0, math.ceil(0.9 * n) - 1)
        p90 = times[p90_idx]
        mean = statistics.fmean(times)
        stats.append(
            TimingStat(
                num_heavy_atoms=h,
                nodes_per_mol=npm,
                samples=n,
                median_s=median,
                p90_s=p90,
                mean_s=mean,
            )
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "num_heavy_atoms": [s.num_heavy_atoms for s in stats],
            "nodes_per_mol": [s.nodes_per_mol for s in stats],
            "samples": [s.samples for s in stats],
            "median_s": [s.median_s for s in stats],
            "p90_s": [s.p90_s for s in stats],
            "mean_s": [s.mean_s for s in stats],
        }
    )
    pq.write_table(table, cache_path)
    logger.info(
        "wrote %s (%d cells from %d records)",
        cache_path,
        len(stats),
        n_records,
    )
    return cache_path


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def load_timing_stats(
    cache_path: Path = DEFAULT_CACHE_PATH,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    auto_rebuild: bool = True,
) -> dict[tuple[int, int], TimingStat]:
    """Load the (heavy_atoms, npm) stats; rebuild if missing.

    Returns ``{(h, npm): TimingStat}``.
    """
    if not cache_path.exists():
        if not auto_rebuild:
            raise FileNotFoundError(cache_path)
        build_timing_cache(runs_root=runs_root, cache_path=cache_path)
    table = pq.read_table(cache_path).to_pylist()
    return {
        (row["num_heavy_atoms"], row["nodes_per_mol"]): TimingStat(**row)
        for row in table
    }


def _per_mol_seconds(
    stats: dict[tuple[int, int], TimingStat],
    heavy: int,
    npm: int,
    *,
    aggregate: str,
) -> tuple[float, str]:
    """Pick a per-mol time for (h, npm).

    Falls back in order: exact (h, npm) → same h any npm scaled by
    ``npm_observed / npm_requested`` (closer to linear strong scaling
    than nothing) → nearest h with the same npm → median of all stats.

    Returns ``(seconds, source_tag)`` where ``source_tag`` is one of
    ``"exact"``, ``"hscale"``, ``"hnearest"``, ``"fallback"``.
    """
    field = {"median": "median_s", "mean": "mean_s", "p90": "p90_s"}[aggregate]

    cell = stats.get((heavy, npm))
    if cell is not None:
        return getattr(cell, field), "exact"

    # Same h, any npm — strong-scale to requested npm (inverse linear, capped).
    same_h = [s for (h, _), s in stats.items() if h == heavy]
    if same_h:
        s = max(same_h, key=lambda x: x.samples)
        # If the historical npm was smaller (more nodes-per-mol = faster),
        # divide; if larger, multiply. Cap the scale factor so we don't
        # extrapolate wildly past observed.
        scale = min(8.0, max(0.125, s.nodes_per_mol / max(1, npm)))
        return getattr(s, field) * scale, "hscale"

    # Nearest h, same npm.
    same_npm = sorted(
        ((abs(h - heavy), s) for (h, n), s in stats.items() if n == npm),
        key=lambda kv: kv[0],
    )
    if same_npm:
        return getattr(same_npm[0][1], field), "hnearest"

    # Last resort: median over everything.
    if stats:
        all_vals = sorted(getattr(s, field) for s in stats.values())
        return all_vals[len(all_vals) // 2], "fallback"

    raise RuntimeError("no timing stats available; rebuild the cache")


def pick_nodes_per_mol(chunk_parquet: Path) -> tuple[int, int]:
    """Pick ``nodes_per_mol`` for a chunk from its largest heavy-atom value.

    Returns ``(npm, max_heavy_atoms)``. Mixed-h chunks are sized to the
    biggest molecule so it gets enough nodes.
    """
    chunk_parquet = Path(chunk_parquet)
    df = pq.read_table(chunk_parquet, columns=["num_heavy_atoms"]).to_pandas()
    if df.empty:
        raise ValueError(f"chunk parquet has zero rows: {chunk_parquet}")
    max_h = int(df["num_heavy_atoms"].max())
    for cap_h, npm in NPM_POLICY:
        if max_h <= cap_h:
            return npm, max_h
    raise RuntimeError("NPM_POLICY has no catch-all entry; fix it")


def pick_queue(chunk_parquet: Path) -> tuple[str, int]:
    """Pick a PBS queue for a chunk from its largest heavy-atom value.

    Returns ``(queue, max_heavy_atoms)``. h≤5 chunks route to ``capacity``
    (168 h walltime cap) so the known CCSD-initialization straggler tail
    on small mols doesn't get walltime-killed; everything else gets the
    routing queue ``prod``.
    """
    chunk_parquet = Path(chunk_parquet)
    df = pq.read_table(chunk_parquet, columns=["num_heavy_atoms"]).to_pandas()
    if df.empty:
        raise ValueError(f"chunk parquet has zero rows: {chunk_parquet}")
    max_h = int(df["num_heavy_atoms"].max())
    for cap_h, queue in QUEUE_POLICY:
        if max_h <= cap_h:
            return queue, max_h
    raise RuntimeError("QUEUE_POLICY has no catch-all entry; fix it")


@dataclass
class Estimate:
    seconds: int
    walltime: str            # HH:MM:SS, clamped to queue cap if provided
    raw_seconds: int         # before clamp
    concurrent_slots: int
    n_mols: int
    aggregate: str           # "median", "mean", or "p90"
    safety: float
    startup_overhead_s: float
    sources: dict[str, int]  # how many mols used each lookup tier
    queue: Optional[str]
    clamped: bool


def estimate_walltime(
    chunk_parquet: Path,
    total_nodes: int,
    nodes_per_mol: int,
    *,
    queue: Optional[str] = None,
    safety: float = DEFAULT_SAFETY_FACTOR,
    startup_overhead_s: float = DEFAULT_STARTUP_OVERHEAD_S,
    aggregate: str = "p90",
    cache_path: Path = DEFAULT_CACHE_PATH,
    runs_root: Path = DEFAULT_RUNS_ROOT,
) -> Estimate:
    """Estimate the PBS walltime needed for a chunk submission.

    ``aggregate`` is one of ``"median"``, ``"mean"``, ``"p90"``. Default
    is ``"p90"`` — the safety factor multiplies on top, so the typical
    behaviour is "be conservative".
    """
    if total_nodes < nodes_per_mol:
        raise ValueError(
            f"total_nodes ({total_nodes}) < nodes_per_mol ({nodes_per_mol})"
        )
    chunk_parquet = Path(chunk_parquet)
    if not chunk_parquet.is_file():
        raise FileNotFoundError(chunk_parquet)

    df = pq.read_table(chunk_parquet, columns=["num_heavy_atoms"]).to_pandas()
    if df.empty:
        raise ValueError(f"chunk parquet has zero rows: {chunk_parquet}")

    stats = load_timing_stats(cache_path, runs_root=runs_root)

    concurrent_slots = total_nodes // nodes_per_mol
    total_serial_s = 0.0
    sources: dict[str, int] = {}
    for h in df["num_heavy_atoms"].astype(int):
        sec, src = _per_mol_seconds(
            stats, h, nodes_per_mol, aggregate=aggregate
        )
        total_serial_s += sec
        sources[src] = sources.get(src, 0) + 1

    # Per-batch wallclock = max-mol-in-batch, not average; we approximate
    # by dividing total serial work by concurrency and multiplying by the
    # safety factor (which absorbs the straggler tail). Add startup once.
    pred_s = total_serial_s / concurrent_slots * safety + startup_overhead_s
    raw_s = int(math.ceil(pred_s))

    cap = QUEUE_WALLTIME_CAPS_S.get(queue) if queue else None
    if cap is not None and raw_s > cap:
        seconds = cap
        clamped = True
    else:
        seconds = raw_s
        clamped = False

    return Estimate(
        seconds=seconds,
        walltime=_seconds_to_hhmmss(seconds),
        raw_seconds=raw_s,
        concurrent_slots=concurrent_slots,
        n_mols=len(df),
        aggregate=aggregate,
        safety=safety,
        startup_overhead_s=startup_overhead_s,
        sources=sources,
        queue=queue,
        clamped=clamped,
    )


def _seconds_to_hhmmss(sec: int) -> str:
    # Round up to whole minutes so PBS sees a clean value.
    sec = int(math.ceil(sec / 60.0) * 60)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# CLI entry-point (`iqc-sweep estimate ...`)
# ---------------------------------------------------------------------------


def cli_main(args) -> int:
    """Subcommand handler installed by cli_sweep._build_parser."""
    if args.rebuild_cache:
        build_timing_cache(runs_root=args.runs_root, cache_path=args.cache_path)
        if not args.chunk:
            print(args.cache_path)
            return 0

    if not args.chunk:
        print("error: --chunk required (or pass --rebuild-cache alone)")
        return 2

    # Resolve --npm: an explicit int, or "auto" → pick from chunk h-class.
    npm_arg = str(args.npm).lower() if args.npm is not None else "auto"
    if npm_arg == "auto":
        npm, max_h = pick_nodes_per_mol(args.chunk)
        npm_source = f"auto (max_h={max_h})"
    else:
        npm = int(args.npm)
        npm_source = "explicit"

    # Resolve --queue: an explicit string, or "auto" → pick from chunk h-class.
    queue_arg = str(args.queue).lower() if args.queue is not None else None
    if queue_arg == "auto":
        queue, _ = pick_queue(args.chunk)
        queue_source = "auto"
    else:
        queue = args.queue  # None or an explicit value
        queue_source = "explicit" if queue else "unset"

    if args.print == "npm":
        print(npm)
        return 0
    if args.print == "queue":
        print(queue or "")
        return 0

    # Walltime estimation requires --nodes.
    if args.nodes is None:
        print("error: --nodes required for walltime estimation", file=sys.stderr)
        return 2

    est = estimate_walltime(
        chunk_parquet=args.chunk,
        total_nodes=args.nodes,
        nodes_per_mol=npm,
        queue=queue,
        safety=args.safety,
        startup_overhead_s=args.startup_overhead_s,
        aggregate=args.aggregate,
        cache_path=args.cache_path,
        runs_root=args.runs_root,
    )

    if args.print == "walltime":
        print(est.walltime)
        return 0
    if args.print == "params":
        # one line, shell-friendly: "<npm> <walltime> <queue>"
        # (queue is "" if neither explicit nor auto-resolved)
        print(f"{npm} {est.walltime} {queue or ''}")
        return 0

    # human report
    print(f"chunk:            {args.chunk}")
    print(f"n_mols:           {est.n_mols}")
    print(f"nodes_per_mol:    {npm}  ({npm_source})")
    print(f"queue:            {queue or '(unset)'}  ({queue_source})")
    print(f"nodes:            {args.nodes}  (npm={npm} → {est.concurrent_slots} concurrent)")
    print(f"aggregate/safety: {est.aggregate} × {est.safety} + {int(est.startup_overhead_s)}s startup")
    print(f"raw estimate:     {_seconds_to_hhmmss(est.raw_seconds)}  ({est.raw_seconds}s)")
    if est.clamped:
        print(f"walltime (clamp): {est.walltime}  [{est.queue} cap]")
    else:
        print(f"walltime:         {est.walltime}")
    if est.sources:
        srcs = ", ".join(f"{k}={v}" for k, v in sorted(est.sources.items()))
        print(f"lookup sources:   {srcs}")
    return 0
