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
    nodes_per_mol: int,
    ppn: int,
    checkpoint_dir: str,
    process_kwargs: dict,
) -> Optional[dict]:
    """Worker entry point. One row → one ``_process_one_row`` call.

    Runs as a serial task (async_loky). For ExaChem calculators, the
    calculator itself submits the binary to the EL cluster via
    ClusterClient (async_mpi) — no manual mpiexec here.

    Returns the result dict, ``None`` (bad input), or the
    ``SKIPPED_EXISTING`` sentinel. The orchestrator side maps each of
    the three outcomes to a counter and an optional JSONL append.
    """

    import os as _os

    _zam = _os.environ.get("ZE_AFFINITY_MASK", "")
    if "." in _zam:
        _os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"

    import ase  # noqa: F401
    import ase.parallel as asepar

    asepar.world = asepar.DummyMPI()

    from iqc.main import _process_one_row as _proc

    from iqc.ensemble_launcher_dispatch import (
        _get_worker_calculator as _get_calc,
        _load_side_cache as _load_cache,
    )

    cp = dict(calc_params)
    cp["el_nnodes"] = nodes_per_mol
    cp["el_ppn"] = ppn

    side = _load_cache(side_cache_path)
    pk = {
        **process_kwargs,
        "xyz_files": side["xyz_files"],
        "completed_file_index": side["completed_file_index"],
        "calc_params": cp,
    }
    partials_dir = pk.pop("partials_dir", None)

    calc = _get_calc(calc_name, cp)

    from ensemble_launcher.orchestrator import ClusterClient

    client = ClusterClient(checkpoint_dir=checkpoint_dir)
    client.start()
    calc.cluster_client = client
    try:
        result = _proc(xyz_index, calculator=calc, **pk)
    finally:
        calc.cluster_client = None
        client.teardown()

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
        num_slots = 1
        head_nodes = all_nodes
    else:
        all_nodes = [node.split(".")[0] for node in _read_pbs_nodes()]
        total_nodes = len(all_nodes)
        num_slots = total_nodes // nodes_per_mol
        if num_slots < 1:
            logging.error(
                "el-nodes-per-mol (%d) exceeds available nodes (%d).",
                nodes_per_mol,
                total_nodes,
            )
            return 1
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

    from ensemble_launcher import EnsembleLauncher
    from ensemble_launcher.config import LauncherConfig, MPIConfig, PolicyConfig, SystemConfig
    from ensemble_launcher.ensemble import Task
    from ensemble_launcher.orchestrator import ClusterClient

    cpus_per_node = int(args.el_cpus_per_node)
    partials_dir = str(Path.cwd() / "results_partials")

    checkpoint_dir = os.path.join(head_output_dir, "el_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    policy_config = PolicyConfig(
        nlevels=nlevels,
        leaf_nodes=num_slots,
    )
    mpi_config = MPIConfig(
        flavor="cray-pals",
        cpu_bind_method="none",
    )

    launcher_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name=["async_loky", "async_mpi"],
        children_scheduler_policy="fixed_leafs_children_policy",
        comm_name="async_zmq",
        cluster=True,
        checkpoint_dir=checkpoint_dir,
        report_interval=args.el_report_interval,
        worker_logs=True,
        master_logs=True,
        return_stdout=False,
        policy_config=policy_config,
        mpi_config=mpi_config,
    )

    system_config = SystemConfig(
        name="aurora",
        ncpus=cpus_per_node,
        ngpus=0,
    )

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=system_config,
        launcher_config=launcher_config,
        Nodes=head_nodes,
        pin_resources=False,
    )

    logging.info(
        "Starting cluster (nlevels=%d, num_slots=%d, leaf_nodes=%d)",
        nlevels,
        num_slots,
        num_slots,
    )
    el.start(wait_time=5)

    raw_results: dict = {}
    task_ids: list[str] = []
    try:
        client = ClusterClient(
            checkpoint_dir=checkpoint_dir, checkpoint_timeout=120.0,
        )
        client.start()
        try:
            futures = {}
            for i in range(number_of_xyz):
                pk = dict(process_kwargs_base)
                pk["worker_id"] = i
                pk["partials_dir"] = partials_dir
                tid = f"row-{i:07d}"
                task_ids.append(tid)
                futures[tid] = client.submit(
                    Task(
                        task_id=tid,
                        nnodes=1,
                        ppn=1,
                        executable=_row_callable,
                        args=(i,),
                        kwargs=dict(
                            calc_name=calculator_name,
                            calc_params=calc_params,
                            side_cache_path=side_cache_path,
                            nodes_per_mol=nodes_per_mol,
                            ppn=ppn,
                            checkpoint_dir=checkpoint_dir,
                            process_kwargs=pk,
                        ),
                        executor_name="async_loky",
                    )
                )

            logging.info(
                "Submitted %d row(s) to cluster", number_of_xyz,
            )

            for tid, fut in futures.items():
                try:
                    raw_results[tid] = fut.result()
                except Exception as e:  # noqa: BLE001
                    raw_results[tid] = e
        finally:
            client.teardown()
    except Exception as e:  # noqa: BLE001
        logging.error("Cluster client error: %s", e, exc_info=True)
    finally:
        el.stop()

    jsonl_file = f"iqc_{task}_results_{run_id}.jsonl"
    completed = 0
    failed = 0
    skipped_existing = 0
    bad_inputs = 0
    with open(jsonl_file, "w") as outfile:
        for tid in task_ids:
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
