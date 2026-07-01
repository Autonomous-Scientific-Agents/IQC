"""Ensemble Launcher dispatch for IQC.

Alternative orchestration path that sits alongside ``iqc.main`` (rank-0
``mpiexec``) and ``iqc.parsl_dispatch`` (Parsl HighThroughputExecutor).
Neither of the other two is modified; pick the dispatcher per job.

The runtime model mirrors ``parsl_dispatch.py`` — persistent Python
workers consume per-row tasks over ZMQ — but the orchestrator is
``ensemble_launcher`` (``/lus/flare/projects/HiFiThermKin/keceli/ensemble_launcher``)
instead of Parsl. That gives a hierarchical master/sub-master/worker
tree proven to 2048+ nodes with no separate interchange broker.

Same import-storm fix as Parsl: the worker pays the iqc + ASE +
calculator load cost ONCE; per-row work after that is a ZMQ message +
``_process_one_row(...)`` + an internal ``mpiexec ExaChem``. The
expensive ``--skip-existing-from`` JSONL walk happens ONCE on the
launcher side, and the result is shipped via a side-cache pickle so
per-task pickles stay small.

CLI is a superset of ``iqc`` — same flags, plus an Ensemble Launcher
group:

    --el-nodes-per-mol N   Aurora: nodes per concurrent ExaChem call.
    --el-nlevels {0,1,2,3} Override hierarchy depth (default: auto).
    --el-ranks-per-mol N   ExaChem ranks per molecule (default 13*npm).
    --el-ppn N             Ranks-per-node hint for the inner mpiexec (default 13).
    --el-local             Single-host smoke test (no $PBS_NODEFILE).

Run as ``iqc-el`` (register in pyproject.toml ``[project.scripts]``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

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


# Module-level worker calculator cache — identical pattern to
# parsl_dispatch._WORKER_CALCULATOR_CACHE. Workers process tasks serially,
# so the dict is single-threaded per worker; no lock needed.
# TODO: dedupe with parsl_dispatch when both modules are stable.
_WORKER_CALCULATOR_CACHE: dict = {}


def _get_worker_calculator(calc_name: str, calc_params: dict):
    """Instantiate (or fetch) the per-worker calculator, cached across calls."""

    from iqc.asetools import get_calculator

    key = (calc_name, json.dumps(calc_params, sort_keys=True, default=str))
    if key not in _WORKER_CALCULATOR_CACHE:
        _WORKER_CALCULATOR_CACHE[key] = get_calculator(name=calc_name, **calc_params)
    return _WORKER_CALCULATOR_CACHE[key]


# Sidecar pickle of xyz_files + completed_file_index. Shipping them in
# every task message would saturate the ZMQ pipe (~60 MB × N tasks).
# Workers load from disk on first task and cache.
_WORKER_SIDE_CACHE: dict = {}


def _load_side_cache(side_cache_path: str) -> dict:
    import pickle as _pickle

    if side_cache_path in _WORKER_SIDE_CACHE:
        return _WORKER_SIDE_CACHE[side_cache_path]
    with open(side_cache_path, "rb") as f:
        data = _pickle.load(f)
    _WORKER_SIDE_CACHE[side_cache_path] = data
    return data


_FAILURE_ERROR_MAX_CHARS = 4096


def _synthesize_failure_row(
    xyz_index: int,
    exception: BaseException,
    *,
    args,
    params_str: str,
    task: str,
    calculator_name: str,
    xyz_files: list,
    input_mode: str,
    number_of_files: int,
) -> dict:
    """Build a result-record dict for a row that raised after exhausting retries.

    Mirrors parsl_dispatch._synthesize_failure_row so the F7 contract
    (failure rows carry the same calculation_key fields as success rows
    and round-trip through the skip-existing index) is preserved.
    """

    from iqc.asetools import (
        atoms2xyz,
        get_atoms_from_smiles,
        get_atoms_from_xyz,
    )

    smiles_input = None
    xyz_record = None
    smiles_record = None
    data_row_index = ""
    if input_mode == "smiles":
        smiles_input = xyz_files[0]
        xyz_file = f"smiles:{smiles_input}"
    elif input_mode == "data_xyz":
        xyz_record = xyz_files[xyz_index]
        xyz_file = f"{args.input}:{args.xyz}[{xyz_record.row_index}]"
        data_row_index = xyz_record.row_index
    elif input_mode == "data_smiles":
        smiles_record = xyz_files[xyz_index]
        smiles_input = smiles_record.smiles
        xyz_file = f"{args.input}:{args.smiles}[{smiles_record.row_index}]"
        data_row_index = smiles_record.row_index
    elif number_of_files > 1:
        xyz_file = xyz_files[xyz_index]
    else:
        xyz_file = xyz_files[0]

    initial_xyz = ""
    try:
        if input_mode == "smiles":
            atoms = get_atoms_from_smiles(smiles_input)
        elif input_mode == "data_xyz":
            atoms = get_atoms_from_xyz(xyz_record.xyz)
        elif input_mode == "data_smiles":
            atoms = get_atoms_from_smiles(smiles_input)
        elif number_of_files > 1:
            atoms = get_atoms_from_xyz(xyz_file)
        else:
            atoms = get_atoms_from_xyz(xyz_file, index=xyz_index)
        initial_xyz = atoms2xyz(atoms)
    except Exception as load_err:  # noqa: BLE001
        logging.warning(
            "Could not reconstruct initial_xyz for failed row %s (%s): %s",
            xyz_index,
            xyz_file,
            load_err,
        )

    err_msg = str(exception)
    if len(err_msg) > _FAILURE_ERROR_MAX_CHARS:
        err_msg = err_msg[:_FAILURE_ERROR_MAX_CHARS] + "...[truncated]"

    return {
        "xyz_file": xyz_file,
        "smiles_input": smiles_input or "",
        "input_mode": input_mode,
        "data_input_file": args.input or "",
        "data_xyz_column": args.xyz if input_mode == "data_xyz" else "",
        "data_smiles_column": args.smiles if input_mode == "data_smiles" else "",
        "data_sort_column": args.sort or "",
        "data_sort_order": args.sort_order if args.sort else "",
        "data_row_index": data_row_index,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": os.uname().nodename,
        "task": task,
        "calculator": calculator_name,
        "model": "",
        "initial_xyz": initial_xyz,
        "params": params_str,
        f"{task}_error": err_msg,
        "el_retries_exhausted": True,
    }


def _row_callable(
    xyz_index: int,
    *,
    calc_name: str,
    calc_params: dict,
    side_cache_path: str,
    slot_node_lists: Optional[list],
    hostfile_tmpdir: Optional[str],
    ranks_per_mol: int,
    ppn: int,
    process_kwargs: dict,
) -> Optional[dict]:
    """Worker entry point. One row → one ``_process_one_row`` call.

    Returns the result dict, ``None`` (bad input), or the
    ``SKIPPED_EXISTING`` sentinel. The orchestrator side maps each of
    the three outcomes to a counter and an optional JSONL append.
    """

    # CRITICAL: Aurora hierarchy fixup must precede any torch import.
    # See parsl_dispatch._row_app for the full reasoning — short version:
    # if Parsl-style "0.0"/"0.1" tile names are in ZE_AFFINITY_MASK, the
    # cluster's default FLAT hierarchy makes them parse to invalid
    # devices. Flip to COMPOSITE only when dot-notation is present.
    import os as _os

    _zam = _os.environ.get("ZE_AFFINITY_MASK", "")
    if "." in _zam:
        _os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"

    import ase  # noqa: F401
    import ase.parallel as asepar

    asepar.world = asepar.DummyMPI()

    from iqc.main import _process_one_row as _proc

    # Re-import sibling helpers (same caveat as parsl_dispatch — the
    # function's pickled __globals__ may not include them on a fresh
    # worker process).
    from iqc.ensemble_launcher_dispatch import (
        _get_worker_calculator as _get_calc,
        _load_side_cache as _load_cache,
    )

    # Self-identify which slot this worker belongs to by matching the
    # local hostname against the per-slot node lists. This avoids the
    # race that pre-baked hostfile paths created: ensemble_launcher's
    # scheduler picks which node a worker lands on independently of any
    # caller-supplied slot ID, so the worker MUST derive its slot at
    # execution time. If no match (single-node / local smoke), fall back
    # to whatever mpi_command calc_params already specifies.
    import socket as _socket
    import uuid as _uuid

    cp = dict(calc_params)
    if slot_node_lists and hostfile_tmpdir:
        my_host = _socket.gethostname()
        my_host_short = my_host.split(".")[0]
        my_slot = None
        for j, nodes in enumerate(slot_node_lists):
            for n in nodes:
                if n == my_host or n.split(".")[0] == my_host_short:
                    my_slot = j
                    break
            if my_slot is not None:
                break
        if my_slot is None:
            logging.warning(
                "_row_callable on host %s could not match any slot; "
                "defaulting to slot 0",
                my_host,
            )
            my_slot = 0
        hf_path = _os.path.join(
            hostfile_tmpdir,
            f"hostfile_slot{my_slot}_{_uuid.uuid4().hex[:8]}.txt",
        )
        with open(hf_path, "w") as _hf:
            _hf.write("\n".join(slot_node_lists[my_slot]) + "\n")
        cp["nproc"] = ranks_per_mol
        cp["mpi_command"] = [
            "mpiexec",
            "--hostfile",
            hf_path,
            "-ppn",
            str(ppn),
            "--cpu-bind=depth",
            "-d",
            "8",
        ]

    side = _load_cache(side_cache_path)
    pk = {
        **process_kwargs,
        "xyz_files": side["xyz_files"],
        "completed_file_index": side["completed_file_index"],
        # The patched calc_params must override what the head sent.
        "calc_params": cp,
    }
    # SIGTERM-graceful partial write: pop partials_dir out of pk before
    # _proc receives it (iqc.main._process_one_row doesn't know about it).
    partials_dir = pk.pop("partials_dir", None)

    calc = _get_calc(calc_name, cp)
    result = _proc(xyz_index, calculator=calc, **pk)

    # Persist the result as a single-line JSONL fragment immediately, so
    # it survives a SIGTERM that kills the EL master before its epilogue
    # runs. The dispatcher's epilogue still writes the consolidated
    # iqc_*_results_*.jsonl from raw_results on clean exit; this is the
    # fallback. _candidate_result_files() globs */results_partials/*.jsonl
    # so the next job's --skip-existing-from sees these and won't re-run.
    if isinstance(result, dict) and partials_dir:
        import json as _json
        from iqc.main import ComplexEncoder as _Enc

        try:
            _os.makedirs(partials_dir, exist_ok=True)
            partial_path = _os.path.join(
                partials_dir, f"row_{xyz_index:07d}.jsonl"
            )
            tmp_path = partial_path + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(_json.dumps(result, cls=_Enc))
                f.write("\n")
            _os.rename(tmp_path, partial_path)
        except Exception as e:  # noqa: BLE001
            logging.warning(
                "row %d: partial write to %s failed: %s",
                xyz_index, partials_dir, e,
            )
    return result


def _add_el_args(parser: argparse.ArgumentParser) -> None:
    """Add Ensemble Launcher options to an existing argparse parser."""

    group = parser.add_argument_group("Ensemble Launcher dispatch options")
    group.add_argument(
        "--el-nodes-per-mol",
        type=int,
        default=1,
        help=(
            "Aurora: nodes per concurrent ExaChem call. K slots = "
            "TOTAL_NODES / el-nodes-per-mol."
        ),
    )
    group.add_argument(
        "--el-ranks-per-mol",
        type=int,
        default=None,
        help=(
            "ExaChem ranks per molecule. Defaults to 13 * el-nodes-per-mol "
            "(matches Aurora PVC: 12 tiles + 1 host rank)."
        ),
    )
    group.add_argument(
        "--el-ppn",
        type=int,
        default=13,
        help="Ranks per node for the slot's inner mpiexec.",
    )
    group.add_argument(
        "--el-nlevels",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help=(
            "Hierarchy depth. Auto if omitted: 1 slot → 0, ≤64 → 1, "
            "≤2048 → 2, larger → 3."
        ),
    )
    group.add_argument(
        "--el-cpus-per-node",
        type=int,
        default=208,
        help=(
            "Logical CPUs per node, declared to the scheduler so each Task "
            "consumes a full node's worth of resources (prevents two ExaChem "
            "instances from co-scheduling on the same node). Aurora default = 208."
        ),
    )
    group.add_argument(
        "--el-report-interval",
        type=float,
        default=30.0,
        help="Seconds between orchestrator status reports.",
    )
    group.add_argument(
        "--el-local",
        action="store_true",
        help=(
            "Single-host smoke test (no $PBS_NODEFILE). Uses the current "
            "hostname as the only node; per-slot hostfile is omitted."
        ),
    )
    group.add_argument(
        "--el-hostfile-dir",
        type=str,
        default=None,
        help=(
            "Directory for per-slot hostfiles. Defaults to "
            "<rundir>/host_chunks (same convention as _split_template.sh)."
        ),
    )


def _parse_args(argv=None) -> argparse.Namespace:
    """Combine the standard IQC parser with Ensemble Launcher extras."""

    iqc_argv = list(sys.argv[1:] if argv is None else argv)
    el_parser = argparse.ArgumentParser(add_help=False)
    _add_el_args(el_parser)
    el_ns, leftover = el_parser.parse_known_args(iqc_argv)
    iqc_ns = _iqc_get_args(leftover)
    for k, v in vars(el_ns).items():
        setattr(iqc_ns, k, v)
    return iqc_ns


def _build_xyz_input_set(args, logger: logging.Logger):
    """Resolve the (xyz_files, input_mode, number_of_xyz, number_of_files)
    tuple from CLI args.

    Verbatim port of parsl_dispatch._build_xyz_input_set so this module
    can stand alone without importing the parsl-dependent module.
    """

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
        logger.info(f"Using XYZ column '{args.xyz}' from data input: {args.input}")
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


class _MakeStaticDir:
    """Picklable callable that returns a fixed directory string and creates it lazily."""

    def __init__(self, path: str):
        self.path = path

    def __call__(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        return self.path


def _read_pbs_nodes() -> list[str]:
    """Read $PBS_NODEFILE into a deduped, order-preserving node list."""

    nf = os.environ.get("PBS_NODEFILE")
    if not nf or not os.path.isfile(nf):
        raise RuntimeError(
            "PBS_NODEFILE not set or unreadable. Use --el-local for a "
            "single-host smoke test, or run from inside a PBS allocation."
        )
    seen: set[str] = set()
    nodes: list[str] = []
    with open(nf) as f:
        for line in f:
            host = line.strip()
            if host and host not in seen:
                seen.add(host)
                nodes.append(host)
    return nodes


def _split_into_hostfiles(
    nodes: list[str], nodes_per_mol: int, hostfile_dir: Path
) -> list[Path]:
    """Write per-slot hostfiles (host_00000 ... host_K-1) and return their paths.

    Mirrors `_split_template.sh` lines 64-67. Returns one Path per slot.
    Tail nodes that don't fill a complete slot are dropped (same as
    `split -l` behavior in the bash version).
    """

    hostfile_dir.mkdir(parents=True, exist_ok=True)
    num_slots = len(nodes) // nodes_per_mol
    paths: list[Path] = []
    for j in range(num_slots):
        chunk = nodes[j * nodes_per_mol : (j + 1) * nodes_per_mol]
        p = hostfile_dir / f"host_{j:05d}"
        p.write_text("\n".join(chunk) + "\n")
        paths.append(p)
    return paths


def _auto_nlevels(num_slots: int) -> int:
    if num_slots <= 1:
        return 0
    if num_slots <= 64:
        return 1
    if num_slots <= 2048:
        return 2
    return 3


def main() -> int:
    """Ensemble Launcher entry point. Returns a shell exit code."""

    start_time = time.time()
    args = _parse_args()

    if args.input and args.input_only:
        from iqc.datatools import run_input_inspection

        return run_input_inspection(args.input)

    input_error = validate_input_args(args)
    if input_error:
        print(input_error, file=sys.stderr)
        return 1

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))
    if not logger.hasHandlers():
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter("IQC %(levelname)s: %(asctime)s [head] %(message)s")
        )
        logger.addHandler(h)

    import ase  # noqa: F401
    import ase.parallel as asepar

    asepar.world = asepar.DummyMPI()

    # Load YAML params (mirrors main.main()).
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

    if (
        str(calculator_name).lower() == "exachem"
        and getattr(args, "keep_artifacts", False)
    ):
        calc_params.setdefault("keep_artifacts", True)

    # Inputs.
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

    completed_file_index: dict = {}
    if args.skip_existing:
        skip_sources = (
            [Path(s).expanduser() for s in args.skip_existing_from]
            if args.skip_existing_from
            else _default_skip_existing_sources()
        )
        include_errors = not getattr(args, "retry_failed_only", False)
        completed_file_index, summary = build_completed_calculation_index(
            skip_sources, include_errors=include_errors
        )
        if db_path or completed_file_index:
            logging.info(
                "Skip-existing enabled: indexed %s completed calculation(s) from "
                "%s record(s) in %s file(s).",
                len(completed_file_index),
                summary["records"],
                summary["files"],
            )

    head_output_dir = _unique_child_path(
        _rank_output_parent(), f"tmp_{task}_el_{run_id}"
    )
    os.makedirs(head_output_dir, exist_ok=True)
    logging.info(f"Per-row work_dir staging area: {head_output_dir}")
    rank_output_dir_factory = _MakeStaticDir(head_output_dir)

    # Decide slot topology.
    nodes_per_mol = max(1, int(args.el_nodes_per_mol))
    ppn = int(args.el_ppn)
    ranks_per_mol = (
        int(args.el_ranks_per_mol) if args.el_ranks_per_mol else ppn * nodes_per_mol
    )

    if args.el_local:
        import socket

        all_nodes = [socket.gethostname()]
        hostfile_paths: list[Optional[Path]] = [None]
        num_slots = 1
        head_nodes = all_nodes
    else:
        all_nodes = _read_pbs_nodes()
        total_nodes = len(all_nodes)
        num_slots = total_nodes // nodes_per_mol
        if num_slots < 1:
            logging.error(
                "el-nodes-per-mol (%d) exceeds available nodes (%d).",
                nodes_per_mol,
                total_nodes,
            )
            return 1
        hostfile_dir = (
            Path(args.el_hostfile_dir)
            if args.el_hostfile_dir
            else Path.cwd() / "host_chunks"
        )
        hostfile_dir.mkdir(parents=True, exist_ok=True)
        # slot_node_lists[j] = list of N=nodes_per_mol nodes assigned to
        # slot j. Workers pick their slot at task time by matching their
        # own hostname (see _row_callable). The pre-baked host_chunks
        # files written below are now diagnostic only — the worker writes
        # its own per-call hostfile at execution time.
        slot_node_lists: list[list[str]] = [
            all_nodes[j * nodes_per_mol : (j + 1) * nodes_per_mol]
            for j in range(num_slots)
        ]
        # Still write diagnostic per-slot hostfiles so the run dir is
        # self-documenting and matches _split_template.sh's layout.
        _split_into_hostfiles(all_nodes, nodes_per_mol, hostfile_dir)
        # Pass ALL nodes (not just slot heads) to ensemble_launcher so
        # it can place workers across the full allocation. nchildren =
        # num_slots in the PolicyConfig below will then place one worker
        # on each slot's first node.
        head_nodes = all_nodes
        logging.info(
            "Topology: total_nodes=%d nodes_per_mol=%d num_slots=%d ranks_per_mol=%d",
            total_nodes,
            nodes_per_mol,
            num_slots,
            ranks_per_mol,
        )

    nlevels = (
        int(args.el_nlevels) if args.el_nlevels is not None else _auto_nlevels(num_slots)
    )

    # Sidecar pickle (xyz_files + skip index) so per-task pickles stay small.
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

    process_kwargs_base = {
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
        "n_workers": number_of_xyz,
        "rank_output_dir_factory": rank_output_dir_factory,
        "direct_work_dir": direct_work_dir,
        "db_path": db_path,
    }

    # Build the Task ensemble. Tasks declare (nnodes=nodes_per_mol,
    # ppn=cpus_per_node) so the scheduler reserves the whole slot per task
    # — preventing two tasks from packing onto the same slot and colliding
    # on the slot's hostfile when each tries to spawn its own ExaChem mpiexec
    # over the same N nodes. Without this, at npm=2 the scheduler packs 2
    # tasks per 2-node worker (each task=1 node), both try mpiexec on the
    # slot's hostfile → mutual deadlock (smoke 8575800: 8 exachem_run_*
    # dirs created, 0 completions in 30 min).
    # Requires the local EL patch in
    # /lus/flare/projects/HiFiThermKin/keceli/ensemble_launcher/ensemble_launcher/
    # executors/async_mp_executor.py allowing multi-node JobResources through
    # AsyncProcessPoolExecutor (runs callable on node 0; caller orchestrates
    # multi-node via inner mpiexec/hostfile). Without that patch every task
    # silently hangs in TaskStatus.RUNNING forever (smoke 8575783: 0 mols
    # completed in 30 min, only a single "MultiProcessingExecutor can only
    # execute single node tasks" line in main.w0.log betrayed it).
    from ensemble_launcher import EnsembleLauncher
    from ensemble_launcher.config import LauncherConfig, MPIConfig, PolicyConfig
    from ensemble_launcher.ensemble import Task

    cpus_per_node = int(args.el_cpus_per_node)
    # SIGTERM-graceful partial result writes — each row callable drops a
    # single-line JSONL fragment here as soon as it finishes, so a walltime
    # SIGTERM that kills the EL master before its epilogue runs doesn't lose
    # the results that were already in raw_results. _candidate_result_files()
    # in iqc.main globs */results_partials/*.jsonl so the next job's
    # --skip-existing-from picks them up just like the consolidated JSONL.
    partials_dir = str(Path.cwd() / "results_partials")
    tasks: dict = {}
    for i in range(number_of_xyz):
        pk = dict(process_kwargs_base)
        pk["worker_id"] = i
        pk["partials_dir"] = partials_dir
        tid = f"row-{i:07d}"
        tasks[tid] = Task(
            task_id=tid,
            nnodes=nodes_per_mol,
            ppn=cpus_per_node,
            executable=_row_callable,
            args=(i,),
            kwargs=dict(
                calc_name=calculator_name,
                calc_params=calc_params,
                side_cache_path=side_cache_path,
                slot_node_lists=(
                    slot_node_lists if not args.el_local else None
                ),
                hostfile_tmpdir=(
                    str(hostfile_dir) if not args.el_local else None
                ),
                ranks_per_mol=ranks_per_mol,
                ppn=ppn,
                process_kwargs=pk,
            ),
        )

    logging.info(
        "Submitting %d row(s) to Ensemble Launcher (nlevels=%d, num_slots=%d)",
        number_of_xyz,
        nlevels,
        num_slots,
    )

    # nlevels and nchildren live INSIDE PolicyConfig in LauncherConfig.
    # Earlier versions of this file passed them as top-level LauncherConfig
    # kwargs — Pydantic silently accepted the extras and the defaults
    # (nchildren=1) won, producing one worker instead of K. nchildren must
    # equal num_slots so SimpleSplitChildrenPolicy spreads workers one-per-
    # slot across the allocation.
    policy_config = PolicyConfig(
        nlevels=nlevels,
        nchildren=num_slots if nlevels > 0 else 1,
        leaf_nodes=nodes_per_mol,
    )
    # child_executor_name="async_mpi" launches each worker via mpiexec
    # so they land on DIFFERENT nodes — required for the per-worker
    # hostname-to-slot identification in _row_callable. The default
    # "async_processpool" spawns workers as local subprocesses of the
    # master, so all workers see the master's hostname and self-identify
    # as the same slot. (Discovered in v3 smoke: both workers wrote
    # hostfile_slot0 because both ran on x4517c2s3b0n0.)
    # MPI config — IMPORTANT: cpu_bind_method="none" so EL does not emit
    # `--cpu-bind list:0-0` for level-2 child sub-master spawns when nlevels>=2
    # (which kicks in above 64 slots in _auto_nlevels). Aurora reserves cpus 0
    # and 52 for system services since 2025-03-31, so PALS rejects list:0-0
    # with "fewer CPUs (0) than depth (1)" — every child spawn fails, cascading
    # into RPC-broken / chdir-denied errors on every compute node (256-node EL
    # run 8574618: 244 MB of worker logs, 0 mols completed). The inner ExaChem
    # mpiexec keeps its own --cpu-bind=depth -d 8 -ppn 13 from
    # sweep_params_1n.yaml (separate code path; unaffected by this flag).
    # flavor="cray-pals" sets the PALS-specific flag names (-n, --ppn,
    # --hostfile) instead of the "mpich" default; on Aurora both PALS and
    # MPICH flavors happen to resolve to the same `mpiexec` binary, but the
    # PALS flavor is the correct semantic mapping.
    mpi_config = MPIConfig(
        flavor="cray-pals",
        cpu_bind_method="none",
    )

    launcher_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        report_interval=args.el_report_interval,
        worker_logs=True,
        master_logs=True,
        return_stdout=False,
        policy_config=policy_config,
        mpi_config=mpi_config,
    )

    # SystemConfig declares each node's CPU capacity so the Task.ppn gate
    # above means "one task per node" rather than "all task ppn against an
    # auto-detected larger count."
    from ensemble_launcher.config import SystemConfig

    system_config = SystemConfig(
        name="aurora",
        ncpus=cpus_per_node,
        ngpus=0,
    )

    el = EnsembleLauncher(
        ensemble_file=tasks,
        system_config=system_config,
        launcher_config=launcher_config,
        Nodes=head_nodes,
        pin_resources=False,  # iqc's inner mpiexec does its own pinning.
    )

    raw_results: dict = {}
    try:
        raw_results_obj = el.run()
    except Exception as e:  # noqa: BLE001 — top-level safety net
        logging.error("Ensemble Launcher run() raised: %s", e, exc_info=True)
        raw_results_obj = None

    # Unwrap EL's return value into {task_id: row_callable_return}. Newer EL
    # versions return a ResultBatch(data=[Result(...), ...]); the Result
    # wrapper has .data (our row_callable return), .success (bool), and
    # .exception (str traceback if .success is False). Older EL versions
    # returned a plain dict already in this shape. Without this unwrapping,
    # `raw_results.get(tid)` raises AttributeError on ResultBatch and the
    # dispatcher loses every result of an otherwise-successful run (smoke
    # 8597150: all 32 mols completed on the compute side, 0 JSONL written
    # because of this very crash).
    if raw_results_obj is None:
        raw_results = {}
    elif hasattr(raw_results_obj, "data") and isinstance(raw_results_obj.data, list):
        for r in raw_results_obj.data:
            if not getattr(r, "success", True) and getattr(r, "exception", None):
                # exception is a str traceback here; wrap in a real Exception
                # so the downstream `isinstance(result, BaseException)` branch
                # routes it to _synthesize_failure_row().
                raw_results[r.task_id] = Exception(r.exception)
            else:
                raw_results[r.task_id] = r.data
    elif isinstance(raw_results_obj, dict):
        raw_results = raw_results_obj
    else:
        logging.error(
            "Ensemble Launcher returned unexpected type %s; treating as empty.",
            type(raw_results_obj),
        )

    # Accounting + JSONL write (mirrors parsl_dispatch's final loop).
    jsonl_file = f"iqc_{task}_results_{run_id}.jsonl"
    completed = 0
    failed = 0
    skipped_existing = 0
    bad_inputs = 0
    with open(jsonl_file, "w") as outfile:
        for tid in tasks:
            row_idx = int(tid.split("-")[1])
            result = raw_results.get(tid)
            if isinstance(result, BaseException):
                failed += 1
                failure_row = _synthesize_failure_row(
                    row_idx,
                    result,
                    args=args,
                    params_str=params_str,
                    task=task,
                    calculator_name=calculator_name,
                    xyz_files=xyz_files,
                    input_mode=input_mode,
                    number_of_files=number_of_files,
                )
                outfile.write(json.dumps(failure_row, cls=ComplexEncoder))
                outfile.write("\n")
                continue
            if result is SKIPPED_EXISTING:
                skipped_existing += 1
                continue
            if result is None:
                bad_inputs += 1
                continue
            if not isinstance(result, dict):
                # Defensive: unexpected return type → log + count as failure
                # without trying to round-trip through the skip-existing
                # index (we don't have the calculation_key fields).
                failed += 1
                logging.error("row %s returned unexpected type %s", tid, type(result))
                continue
            result.pop("_unique_name", None)
            result.pop("_record_stamp", None)
            result.pop("_work_dir_used", None)
            outfile.write(json.dumps(result, cls=ComplexEncoder))
            outfile.write("\n")
            completed += 1

    logging.info(
        "Ensemble Launcher dispatch complete: %s completed, %s skipped-existing, "
        "%s bad-input, %s failed.",
        completed,
        skipped_existing,
        bad_inputs,
        failed,
    )

    if completed > 0 or failed > 0:
        try:
            convert_jsonl_results_to_parquet(jsonl_file)
        except ValueError as e:
            logging.warning(f"Skipping parquet conversion: {e}")
        except Exception as e:  # noqa: BLE001
            logging.error(
                f"Failed to convert JSONL results to parquet: {e}", exc_info=True
            )

    if db_path:
        from iqc.main import insert_jsonl_to_db

        logging.info(f"Importing results from {jsonl_file} into {db_path}")
        insert_jsonl_to_db(jsonl_file, db_path)

    logging.info(f"Total time: {time.time() - start_time:.2f}s")
    return 0 if failed == 0 else 2


def run_cli() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(run_cli())
