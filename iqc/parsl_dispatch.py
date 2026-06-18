"""Parsl-based dispatch for IQC.

Replaces the raw ``mpiexec`` per-rank dispatch with a Parsl ``HighThroughputExecutor``:
the head process iterates input rows and submits one ``@python_app`` per row,
Parsl pins each worker to its own Intel GPU tile, and the per-row work runs
inside reusable workers. When a worker dies (e.g. Level Zero GPU abort), Parsl
re-queues the failed row on a healthy worker via ``retries`` instead of taking
the whole job down with it.

CLI is a superset of ``iqc`` — same flags, plus a Parsl-specific group:

    --parsl-local          Use LocalProvider (no PBS), good for smoke tests.
    --parsl-workers N      Worker count for --parsl-local.
    --parsl-retries N      Override the default retry count.
    --parsl-nodes N        Aurora: PBS nodes per block (default 1).
    --parsl-queue Q        Aurora: PBS queue (default debug).
    --parsl-walltime H:MM:SS  Aurora: walltime.

Run as ``iqc-parsl`` (see pyproject.toml ``[project.scripts]``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Optional

import yaml

import parsl
from parsl import python_app

from iqc.cli import get_args as _iqc_get_args
from iqc.databasetools import create_database
from iqc.datatools import read_smiles_column_records, read_xyz_column_records
from iqc.main import (
    ComplexEncoder,
    SKIPPED_EXISTING,
    _default_skip_existing_sources,
    _make_run_id,
    _rank_output_parent,
    _unique_child_path,
    build_completed_calculation_index,
    convert_jsonl_results_to_parquet,
    get_structure_input_mode,
    validate_input_args,
)


# Module-level worker cache. Each Parsl worker process is long-lived: the first
# row pays the calculator-load cost (~30s for MACE on XPU), subsequent rows
# reuse the cached instance. Keyed on (name, frozenset(params.items())).
_WORKER_CALCULATOR_CACHE: dict = {}


def _get_worker_calculator(calc_name: str, calc_params: dict):
    """Instantiate (or fetch) the per-worker calculator, cached across calls.

    HighThroughputExecutor workers process tasks serially, so this dict is
    accessed from a single thread per worker — no lock needed.
    """

    from iqc.asetools import get_calculator

    # json.dumps with default=str tolerates lists/dicts/other unhashables that
    # may appear in calc_params (e.g. nuclei lists for NMR).
    key = (calc_name, json.dumps(calc_params, sort_keys=True, default=str))
    if key not in _WORKER_CALCULATOR_CACHE:
        _WORKER_CALCULATOR_CACHE[key] = get_calculator(name=calc_name, **calc_params)
    return _WORKER_CALCULATOR_CACHE[key]


# Module-level singletons for things workers should load ONCE from disk, not
# receive in every task pickle. The original design shipped the full xyz_files
# list (~58k entries) and completed_file_index (~19k entries) inside every task
# dict — at 1MB+ per task, this saturated the interchange and slowed task
# submission to ~2 tasks/sec.
_WORKER_SIDE_CACHE: dict = {}


def _load_side_cache(side_cache_path: str) -> dict:
    """Load (and cache) the driver-written sidecar pickle on this worker."""

    import pickle as _pickle

    if side_cache_path in _WORKER_SIDE_CACHE:
        return _WORKER_SIDE_CACHE[side_cache_path]
    with open(side_cache_path, "rb") as f:
        data = _pickle.load(f)
    _WORKER_SIDE_CACHE[side_cache_path] = data
    return data


@python_app
def _row_app(
    xyz_index: int,
    *,
    calc_name: str,
    calc_params: dict,
    side_cache_path: str,
    process_kwargs: dict,
) -> Optional[dict]:
    """Parsl app: process one input row. Returns the result dict, or None if skipped."""

    # ASE before any MPI-aware imports (mirrors the comment in iqc.main.main()).
    import ase  # noqa: F401
    import ase.parallel as asepar

    asepar.world = asepar.DummyMPI()

    from iqc.main import _process_one_row as _proc

    # Re-import the cache helpers from this module rather than relying on the
    # @python_app's pickled __globals__: Parsl reconstructs the function on
    # the worker with a restricted globals dict that does NOT include sibling
    # module-level helpers, so a bare `_get_worker_calculator(...)` raises
    # NameError the moment a row is dispatched.
    from iqc.parsl_dispatch import (
        _get_worker_calculator as _get_calc,
        _load_side_cache as _load_cache,
    )

    side = _load_cache(side_cache_path)
    # The driver wrote {xyz_files, completed_file_index} once to disk; merge
    # them into the per-task kwargs here so _process_one_row sees them.
    process_kwargs = {
        **process_kwargs,
        "xyz_files": side["xyz_files"],
        "completed_file_index": side["completed_file_index"],
    }

    calc = _get_calc(calc_name, calc_params)
    return _proc(
        xyz_index,
        calculator=calc,
        **process_kwargs,
    )


def _add_parsl_args(parser: argparse.ArgumentParser) -> None:
    """Add Parsl-specific options to an existing argparse parser."""

    group = parser.add_argument_group("Parsl dispatch options")
    group.add_argument(
        "--parsl-local",
        action="store_true",
        help=(
            "Use a LocalProvider HighThroughputExecutor instead of submitting "
            "via PBS. Good for smoke tests on a single compute node."
        ),
    )
    group.add_argument(
        "--parsl-workers",
        type=int,
        default=12,
        help=(
            "Number of concurrent workers when --parsl-local is set. Defaults "
            "to 12 (the Aurora tile count per node)."
        ),
    )
    group.add_argument(
        "--parsl-retries",
        type=int,
        default=2,
        help="Retry count for each row when a worker fails.",
    )
    group.add_argument(
        "--parsl-nodes",
        type=int,
        default=1,
        help="Aurora PBS: nodes per block.",
    )
    group.add_argument(
        "--parsl-queue",
        type=str,
        default="debug",
        help="Aurora PBS: queue name (debug, debug-scaling, prod, ...).",
    )
    group.add_argument(
        "--parsl-walltime",
        type=str,
        default="0:30:00",
        help="Aurora PBS: walltime, format H:MM:SS.",
    )
    group.add_argument(
        "--parsl-account",
        type=str,
        default="IQC",
        help="Aurora PBS: charging account.",
    )
    group.add_argument(
        "--parsl-venv-activate",
        type=str,
        default=None,
        help=(
            "Aurora PBS: absolute path to the venv's bin/activate. Defaults to "
            "the venv that contains the running interpreter."
        ),
    )
    group.add_argument(
        "--parsl-tile-pin",
        action="store_true",
        help=(
            "Local mode only: pin workers to Aurora GPU tiles via "
            "available_accelerators (12 tiles)."
        ),
    )
    group.add_argument(
        "--parsl-single-alloc",
        action="store_true",
        help=(
            "Run inside the current PBS allocation (LocalProvider + "
            "MpiExecLauncher --ppn 1). One qsub for driver+workers — no "
            "separate worker PBS job, no driver/worker queue race. Use this "
            "when launching iqc-parsl from a PBS submit script that already "
            "owns the worker nodes."
        ),
    )


def _parse_args(argv=None) -> argparse.Namespace:
    """Parse args by combining the standard IQC parser with Parsl extras."""

    # Build the parser the same way iqc.cli.get_args does, then add our group.
    # The simplest correct approach: pre-parse with iqc.cli, then layer our
    # Parsl flags on top by parsing twice. Cleaner: just add to a fresh parser
    # that wraps the IQC one.
    iqc_argv = list(sys.argv[1:] if argv is None else argv)
    # Pull our --parsl-* flags out before handing off to iqc's get_args, since
    # iqc.cli.get_args would reject unknown flags.
    parsl_parser = argparse.ArgumentParser(add_help=False)
    _add_parsl_args(parsl_parser)
    parsl_ns, leftover = parsl_parser.parse_known_args(iqc_argv)
    # Defer to iqc.cli for the rest.
    iqc_ns = _iqc_get_args(leftover)
    # Merge: copy parsl fields onto the iqc namespace.
    for k, v in vars(parsl_ns).items():
        setattr(iqc_ns, k, v)
    return iqc_ns


def _default_venv_activate() -> str:
    """Best-effort guess at the venv activate script for the running interpreter."""

    return os.path.join(sys.prefix, "bin", "activate")


def _build_xyz_input_set(args, logger: logging.Logger):
    """Mirror main.main()'s input-resolution block. Returns (xyz_files, input_mode, number_of_xyz, number_of_files)."""

    from iqc.asetools import get_atoms_from_smiles  # noqa: F401 — sanity import
    from iqc.xyztools import count_xyz_frames

    input_mode = get_structure_input_mode(args)

    if input_mode == "smiles":
        xyz_files = [args.smiles]
        number_of_xyz = 1
        number_of_files = 1
        logger.info(f"Using SMILES input: {args.smiles}")
        return xyz_files, input_mode, number_of_xyz, number_of_files

    if input_mode == "data_xyz":
        xyz_files = read_xyz_column_records(
            args.input,
            args.xyz,
            sort_column=args.sort,
            sort_order=args.sort_order,
        )
        number_of_xyz = len(xyz_files)
        number_of_files = number_of_xyz
        logger.info(
            f"Using XYZ column '{args.xyz}' from data input: {args.input}"
        )
        if args.sort:
            logger.info(
                f"Sorted data input by column '{args.sort}' ({args.sort_order})."
            )
        logger.info(f"Number of configurations: {number_of_xyz}")
        return xyz_files, input_mode, number_of_xyz, number_of_files

    if input_mode == "data_smiles":
        xyz_files = read_smiles_column_records(
            args.input,
            args.smiles,
            sort_column=args.sort,
            sort_order=args.sort_order,
        )
        number_of_xyz = len(xyz_files)
        number_of_files = number_of_xyz
        logger.info(
            f"Using SMILES column '{args.smiles}' from data input: {args.input}"
        )
        if args.sort:
            logger.info(
                f"Sorted data input by column '{args.sort}' ({args.sort_order})."
            )
        logger.info(f"Number of configurations: {number_of_xyz}")
        return xyz_files, input_mode, number_of_xyz, number_of_files

    # File-system XYZ input
    import glob

    if os.path.isdir(args.xyz):
        xyz_dir = args.xyz
        xyz_files = sorted(glob.glob(os.path.join(xyz_dir, "*.xyz")))
        number_of_xyz = len(xyz_files)
        number_of_files = number_of_xyz
    elif os.path.isfile(args.xyz):
        xyz_files = [args.xyz]
        number_of_xyz = count_xyz_frames(args.xyz)
        number_of_files = 1
    else:
        raise FileNotFoundError(
            f"Input path {args.xyz} does not exist or is not a file/directory."
        )

    if number_of_xyz == 0:
        raise ValueError(f"No .xyz files found in {args.xyz}.")
    logger.info(f"Found {number_of_files} .xyz file(s).")
    logger.info(f"Number of configurations: {number_of_xyz}")
    return xyz_files, input_mode, number_of_xyz, number_of_files


def _build_parsl_config(args):
    """Build the Parsl Config for this run from CLI args."""

    from iqc.parsl_config import (
        AURORA_TILE_NAMES,
        make_aurora_config,
        make_aurora_single_alloc_config,
        make_local_config,
    )

    if args.parsl_single_alloc:
        # Driver runs inside the PBS allocation; LocalProvider + MpiExecLauncher
        # spreads workers across all $PBS_NODEFILE nodes. No separate qsub for
        # workers, so no driver/worker queue race. The node count MUST equal
        # the PBS allocation — without it the launcher's `mpiexec -n` defaults
        # to 1 and only one manager spawns regardless of how many nodes are
        # allocated. Always derive from $PBS_NODEFILE for correctness.
        nodes_file = os.environ.get("PBS_NODEFILE")
        if nodes_file and os.path.isfile(nodes_file):
            with open(nodes_file) as f:
                nodes = sum(1 for line in f if line.strip())
        else:
            nodes = 1
            logging.warning(
                "--parsl-single-alloc but no PBS_NODEFILE — falling back to "
                "1 node (driver-host only)."
            )
        logging.info(f"single-alloc parsl: using {nodes} nodes from PBS_NODEFILE")
        return make_aurora_single_alloc_config(
            nodes_per_block=nodes,
            retries=args.parsl_retries,
        )

    if args.parsl_local:
        accel = AURORA_TILE_NAMES if args.parsl_tile_pin else None
        return make_local_config(
            max_workers=args.parsl_workers,
            available_accelerators=accel,
            retries=args.parsl_retries,
        )

    venv_activate = args.parsl_venv_activate or _default_venv_activate()
    return make_aurora_config(
        venv_activate=venv_activate,
        nodes_per_block=args.parsl_nodes,
        max_blocks=1,
        queue=args.parsl_queue,
        walltime=args.parsl_walltime,
        account=args.parsl_account,
        execute_dir=os.getcwd(),
        retries=args.parsl_retries,
    )


def main() -> int:
    """Parsl entry point. Returns a shell exit code."""

    start_time = time.time()
    args = _parse_args()
    if args.input and args.input_only:
        from iqc.datatools import run_input_inspection

        return run_input_inspection(args.input)
    input_error = validate_input_args(args)
    if input_error:
        print(input_error, file=sys.stderr)
        return 1

    # Logging setup (head process).
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))
    if not logger.hasHandlers():
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("IQC %(levelname)s: %(asctime)s [head] %(message)s"))
        logger.addHandler(h)

    # Mirror iqc.main: ASE before anything else that might pull MPI.
    import ase  # noqa: F401
    import ase.parallel as asepar

    asepar.world = asepar.DummyMPI()

    # Load YAML params (matches main.main()).
    params: dict = {}
    params_str = ""
    if args.params and os.path.isfile(args.params):
        with open(args.params, "r") as f:
            params_str = f.read()
            params = yaml.safe_load(params_str) or {}
        logging.info(f"Loaded parameters from {args.params}")
    elif args.params:
        logging.warning(
            f"Parameter file specified ({args.params}) but not found. Using defaults."
        )
    calc_params = params.get("calculator_params", {})
    opt_params = params.get("optimization_params", {})
    vib_params = params.get("vibration_params", {})
    ir_params = params.get("ir_params", {})
    thermo_params = params.get("thermo_params", {})
    nmr_params = params.get("nmr_params", {})
    vib_params = {**opt_params, **vib_params}
    ir_params = {**vib_params, **ir_params}
    thermo_params = {**vib_params, **thermo_params}
    logging.info(f"All parameters: {params}")

    task = args.task
    calculator_name = (
        args.calculator if args.calculator is not None else params.get("calculator", "mace")
    )
    if task == "nmr":
        calculator_name = args.backend or nmr_params.get("backend", "orca")

    # Resolve inputs.
    try:
        xyz_files, input_mode, number_of_xyz, number_of_files = _build_xyz_input_set(
            args, logger
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error(str(e))
        return 1

    run_id = _make_run_id()
    direct_work_dir = _unique_child_path(args.scratch, f"iqc_{task}_{run_id}")

    db_path = args.database
    if db_path:
        logging.info(f"Checking/Creating database at: {db_path}")
        create_database(db_path)

    completed_file_index: set = set()
    if args.skip_existing:
        skip_sources = (
            [Path(s).expanduser() for s in args.skip_existing_from]
            if args.skip_existing_from
            else _default_skip_existing_sources()
        )
        completed_file_index, summary = build_completed_calculation_index(skip_sources)
        if db_path or completed_file_index:
            logging.info(
                "Skip-existing enabled: indexed %s completed calculation(s) from "
                "%s record(s) in %s file(s).",
                len(completed_file_index),
                summary["records"],
                summary["files"],
            )

    # Pre-create the head's output directory (used as the worker rank_output_dir
    # factory's destination — in the Parsl prototype, all rows share one dir).
    head_output_dir = _unique_child_path(
        _rank_output_parent(), f"tmp_{task}_parsl_{run_id}"
    )
    os.makedirs(head_output_dir, exist_ok=True)
    logging.info(f"Per-row work_dir staging area: {head_output_dir}")

    # Worker-side dir factory (closes over the precreated path — no nonlocal,
    # so it pickles cleanly into the @python_app args).
    rank_output_dir_factory = _MakeStaticDir(head_output_dir)

    # Build the Parsl config and load.
    cfg = _build_parsl_config(args)
    parsl.load(cfg)
    logging.info(
        "Parsl loaded: executor=%s, retries=%s",
        cfg.executors[0].label,
        cfg.retries,
    )

    # Write xyz_files + completed_file_index ONCE to a sidecar pickle on disk.
    # Workers load this on their first task and cache it (see _load_side_cache
    # in this module). The per-task pickle then stays ~10 KB instead of ~60 MB
    # when 58k-row parquet input + 19k-entry skip index would otherwise ship in
    # every TasksOutgoing message.
    import pickle as _pickle

    side_cache_path = os.path.join(head_output_dir, f"side_cache_{run_id}.pkl")
    with open(side_cache_path, "wb") as _f:
        _pickle.dump(
            {
                "xyz_files": xyz_files,
                "completed_file_index": completed_file_index,
            },
            _f,
            protocol=_pickle.HIGHEST_PROTOCOL,
        )
    logging.info(f"Wrote sidecar cache: {side_cache_path}")

    # Build the per-row kwargs that go into every @python_app call. The
    # calculator itself is NOT shipped — each worker builds its own via
    # _get_worker_calculator (cached). Likewise xyz_files and the skip index
    # are loaded from side_cache_path on the worker, not shipped per task.
    process_kwargs = {
        "args": args,
        "params_str": params_str,
        "task": task,
        "calculator_name": calculator_name,
        "calc_params": calc_params,
        "opt_params": opt_params,
        "vib_params": vib_params,
        "ir_params": ir_params,
        "thermo_params": thermo_params,
        "nmr_params": nmr_params,
        "input_mode": input_mode,
        "number_of_files": number_of_files,
        "worker_id": 0,
        "n_workers": 0,
        "rank_output_dir_factory": rank_output_dir_factory,
        "direct_work_dir": direct_work_dir,
        "db_path": db_path,
    }

    logging.info(f"Submitting {number_of_xyz} row(s) to Parsl...")
    futures = []
    for i in range(number_of_xyz):
        # Stamp worker_id with the row index so the result dict has unique IDs.
        # (In production Parsl mode we don't have a stable rank-like number per
        # worker, so we just use xyz_index — it still uniquifies output names.)
        pk = dict(process_kwargs)
        pk["worker_id"] = i
        pk["n_workers"] = number_of_xyz
        futures.append(
            _row_app(
                i,
                calc_name=calculator_name,
                calc_params=calc_params,
                side_cache_path=side_cache_path,
                process_kwargs=pk,
            )
        )

    jsonl_file = f"iqc_{task}_results_{run_id}.jsonl"
    completed = 0
    failed = 0
    skipped_existing = 0
    bad_inputs = 0
    try:
        with open(jsonl_file, "w") as outfile:
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception as e:
                    failed += 1
                    logging.error("row failed after retries: %s", e)
                    continue
                if result is SKIPPED_EXISTING:
                    skipped_existing += 1
                    continue
                if result is None:
                    bad_inputs += 1
                    continue
                # Strip helper keys before serializing — they're caller-only.
                result.pop("_unique_name", None)
                result.pop("_record_stamp", None)
                result.pop("_work_dir_used", None)
                outfile.write(json.dumps(result, cls=ComplexEncoder))
                outfile.write("\n")
                outfile.flush()
                completed += 1
        logging.info(
            "Parsl dispatch complete: %s completed, %s skipped-existing, "
            "%s bad-input, %s failed.",
            completed,
            skipped_existing,
            bad_inputs,
            failed,
        )
    finally:
        # Always cleanup the DFK, even on KeyboardInterrupt or unexpected
        # exceptions — otherwise the HTEX process manager and worker pool
        # stay stranded on the compute node.
        try:
            parsl.dfk().cleanup()
        except Exception as e:
            logging.warning("Parsl DFK cleanup raised: %s", e)

    if args.direct_db and db_path:
        logging.info(
            "Note: --direct-db is not yet supported in Parsl dispatch; results "
            "were written to JSONL (%s) and will be imported below.",
            jsonl_file,
        )

    if completed > 0:
        try:
            convert_jsonl_results_to_parquet(jsonl_file)
        except ValueError as e:
            logging.warning(f"Skipping parquet conversion: {e}")
        except Exception as e:
            logging.error(
                f"Failed to convert JSONL results to parquet: {e}", exc_info=True
            )

    if db_path:
        from iqc.main import insert_jsonl_to_db

        logging.info(f"Importing results from {jsonl_file} into {db_path}")
        insert_jsonl_to_db(jsonl_file, db_path)

    logging.info(f"Total time: {time.time() - start_time:.2f}s")
    return 0 if failed == 0 else 2


class _MakeStaticDir:
    """Picklable callable that returns a fixed directory string and creates it lazily."""

    def __init__(self, path: str):
        self.path = path

    def __call__(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        return self.path


def run_cli() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(run_cli())
