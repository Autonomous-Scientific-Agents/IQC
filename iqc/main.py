import glob
import json
import logging
import os
import pickle
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml  # Import YAML

from iqc.cli import get_args
from iqc.databasetools import (
    calculation_exists,
    calculation_key,
    calculation_key_from_record,
    create_database,
    insert_entries,
    insert_entry,
)
from iqc.datatools import (
    read_smiles_column_records,
    read_xyz_column_records,
    run_input_inspection,
)


def _smiles_to_basename(smiles: str) -> str:
    """Convert a SMILES string into a filesystem-friendly stem."""

    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(smiles)).strip("_")
    return safe[:32] or "smiles"


def _make_run_id():
    """Return a readable, collision-resistant identifier for this IQC run."""

    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _unique_child_path(parent, child_name):
    """Return an absolute child path that does not already exist."""

    parent_path = Path(parent).expanduser()
    candidate = parent_path / child_name
    counter = 1
    while candidate.exists():
        candidate = parent_path / f"{child_name}_{counter}"
        counter += 1
    return os.path.abspath(os.path.expanduser(str(candidate)))


def _rank_output_parent(cwd="."):
    """Return the directory that contains per-rank tmp_* result directories."""

    return Path(cwd).expanduser() / "tmp"


def _default_skip_existing_sources(cwd="."):
    """Return default IQC result files to scan for completed calculations."""

    root = Path(cwd).expanduser()
    sources = list(root.glob("iqc_*_results_*.jsonl"))
    tmp_parents = [root, _rank_output_parent(root)]
    for tmp_parent in tmp_parents:
        if not tmp_parent.is_dir():
            continue
        for tmp_dir in tmp_parent.glob("tmp_*"):
            if tmp_dir.is_dir():
                sources.extend(tmp_dir.glob("*.json"))
    return sorted(set(sources))


def _candidate_result_files(source):
    """Return JSON/JSONL result files under a file or directory source."""

    path = Path(source).expanduser()
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.suffix.lower() in {".json", ".jsonl"} else []

    files = []
    for pattern in ("*.json", "*.jsonl"):
        files.extend(path.rglob(pattern))
    return sorted(set(files))


def _iter_result_records(result_file):
    """Yield dict records from an IQC JSON or JSONL result file."""

    suffix = result_file.suffix.lower()
    try:
        with open(result_file, "r") as handle:
            if suffix == ".jsonl":
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        yield None
                        continue
                    yield record if isinstance(record, dict) else None
            else:
                try:
                    data = json.load(handle)
                except json.JSONDecodeError:
                    yield None
                    return
                if isinstance(data, dict):
                    yield data
                elif isinstance(data, list):
                    for record in data:
                        yield record if isinstance(record, dict) else None
                else:
                    yield None
    except OSError:
        yield None


def _record_status(record):
    """Classify an IQC result record as 'ok' or 'error' based on {task}_error keys."""

    # A record is considered an error if it has a non-empty <task>_error key.
    # F7 also marks Parsl terminal failures with parsl_retries_exhausted=True.
    task = record.get("task")
    if task:
        err_key = f"{task}_error"
        err_val = record.get(err_key)
        if err_val:
            return "error"
    # Fallback: scan for any *_error key with truthy value (defensive against
    # records written before a task field was finalized).
    for key, value in record.items():
        if isinstance(key, str) and key.endswith("_error") and value:
            return "error"
    if record.get("parsl_retries_exhausted"):
        return "error"
    return "ok"


def build_completed_calculation_index(sources, include_errors=True):
    """Map calculation keys to 'ok'/'error' so callers can skip selectively."""

    index = {}
    summary = {
        "sources": len(sources),
        "files": 0,
        "records": 0,
        "invalid": 0,
        "ok": 0,
        "error": 0,
    }
    seen_files = set()

    for source in sources:
        for result_file in _candidate_result_files(source):
            if result_file in seen_files:
                continue
            seen_files.add(result_file)
            summary["files"] += 1
            for record in _iter_result_records(result_file):
                if record is None:
                    summary["invalid"] += 1
                    continue
                try:
                    key = calculation_key_from_record(record)
                except (TypeError, ValueError):
                    summary["invalid"] += 1
                    continue
                status = _record_status(record)
                summary["records"] += 1
                summary[status] += 1
                if status == "error" and not include_errors:
                    continue
                # Prefer 'ok' over 'error' when both records exist for the same
                # key (a later successful retry should win over an earlier failure).
                existing = index.get(key)
                if existing == "ok":
                    continue
                index[key] = status

    return index, summary


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(
            obj,
            (
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        # Handle np.int_ (removed in NumPy 2.0)
        try:
            if isinstance(obj, np.int_):
                return int(obj)
        except (AttributeError, TypeError):
            pass
        # Handle float types
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        # Handle np.float_ (removed in NumPy 2.0)
        try:
            if isinstance(obj, np.float_):
                return float(obj)
        except (AttributeError, TypeError):
            pass
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle np.bool_ (removed in NumPy 2.0)
        try:
            if isinstance(obj, np.bool_):
                return bool(obj)
        except (AttributeError, TypeError):
            pass
        if hasattr(obj, "item"):  # Handle other numpy types
            return obj.item()
        return super().default(obj)


def save_results(results, output_file):
    """Save results to file with fallback options."""
    try:
        # First attempt: Save as JSON
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, cls=ComplexEncoder)
        logging.info(f"Results saved to {output_file} in JSON format")
    except (TypeError, ValueError) as e:
        logging.warning(f"JSON serialization failed: {e}. Trying pickle...")
        try:
            # Second attempt: Save as pickle
            pickle_file = output_file.replace(".json", ".pkl")
            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)
            logging.info(f"Results saved to {pickle_file} in pickle format")
        except Exception as e:
            logging.warning(f"Pickle serialization failed: {e}. Saving as text...")
            # Third attempt: Save as text
            txt_file = output_file.replace(".json", ".txt")
            with open(txt_file, "w") as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
            logging.info(f"Results saved to {txt_file} in text format")


def insert_jsonl_to_db(jsonl_file, db_path):
    with open(jsonl_file, "r") as f:
        summary = insert_entries(f, db_path)
    logging.info(
        "Database import complete: %s processed, %s inserted, %s duplicates skipped.",
        summary["processed"],
        summary["inserted"],
        summary["duplicates"],
    )


def convert_jsonl_results_to_parquet(jsonl_file):
    """Convert a completed IQC JSONL result file to parquet."""

    from scripts.jsonl2parquet import (
        convert_jsonl_to_parquet,
        infer_output_path,
    )

    jsonl_path = Path(jsonl_file)
    parquet_path = infer_output_path(jsonl_path)
    convert_jsonl_to_parquet(str(jsonl_path), str(parquet_path))
    logging.info(f"Parquet results saved to {parquet_path}")
    return parquet_path


def get_nmr_cli_overrides(args):
    """Collect NMR-specific CLI overrides."""

    mapping = {
        "backend": "backend",
        "optimization_backend": "optimization_backend",
        "nuclei": "nuclei",
        "method": "method",
        "basis": "basis",
        "optimization_method": "optimization_method",
        "optimization_basis": "optimization_basis",
        "solvent_model": "solvent_model",
        "solvent": "solvent",
        "charge": "charge",
        "multiplicity": "multiplicity",
        "optimize_geometry": "optimize_geometry",
        "conformer_sampling": "conformer_sampling",
        "num_conformers": "num_conformers",
        "temperature": "temperature",
        "linewidth": "linewidth",
        "lineshape": "lineshape",
        "plot_range": "plot_range",
        "output_dir": "output_dir",
        "reference_shielding": "reference_shieldings",
    }
    overrides = {}
    for arg_name, param_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[param_name] = value
    return overrides


def validate_input_args(args):
    """Return a user-facing validation error for unsupported --input combinations."""

    if args.sort and not args.input:
        return "Error: --sort can only be used with --input."
    if args.sort_order_explicit and not args.sort:
        return "Error: --sort_order requires --sort COLUMN."
    if args.direct_db and not args.database:
        return "Error: --direct-db requires --database DB_PATH."
    if args.skip_existing_from and not args.skip_existing:
        return "Error: --skip-existing-from requires --skip-existing."
    if not args.input and not args.smiles:
        xyz_path = Path(str(args.xyz))
        if xyz_path.suffix.lower() in {
            ".json",
            ".jsonl",
            ".parquet",
            ".csv",
            ".tsv",
            ".xlsx",
            ".xls",
            ".feather",
            ".arrow",
            ".ipc",
        }:
            return (
                f"Error: {args.xyz!r} looks like a tabular/result file, not an "
                "XYZ file. Use --input FILE for JSONL/parquet/CSV inputs. For "
                "IQC result files, omit --xyz to use the opt_xyz column or pass "
                "--xyz COLUMN explicitly."
            )
    if not args.input or args.input_only:
        return None
    if args.input_xyz_column and args.input_smiles_column:
        return (
            "Error: --input can use either --xyz COLUMN or --smiles COLUMN, "
            "but not both."
        )
    if not args.input_xyz_column and not args.input_smiles_column:
        return (
            "Error: when using --input for calculations, pass --xyz COLUMN "
            "or --smiles COLUMN to identify the structure column. If --xyz "
            "is omitted, IQC defaults to the opt_xyz column."
        )
    return None


def get_structure_input_mode(args):
    """Return the molecular input mode implied by parsed CLI arguments."""

    if args.input and args.input_xyz_column:
        return "data_xyz"
    if args.input and args.input_smiles_column:
        return "data_smiles"
    if args.smiles:
        return "smiles"
    return "xyz"


class _SkipExisting:
    """Sentinel: ``_process_one_row`` returns this when the row matched a
    skip-existing key. Lets the caller distinguish it from a ``None`` return
    (which means the input could not be read) so ``skipped_existing`` only
    counts the cases the user asked about with ``--skip-existing``.

    ``__reduce__`` keeps the pickle round-trip identity-preserving so the
    Parsl driver's ``result is SKIPPED_EXISTING`` check survives the trip
    back from a worker process. Without it, every worker returns a fresh
    ``_SkipExisting()`` instance, ``is`` returns False, and the dispatcher
    falls through to ``result.pop("_unique_name", None)`` which raises
    ``AttributeError: '_SkipExisting' object has no attribute 'pop'``.
    """

    __slots__ = ()

    def __repr__(self):
        return "<SKIPPED_EXISTING>"

    def __reduce__(self):
        # Tell pickle to recover the same module-level singleton on unpickle.
        return (_get_skipped_existing_singleton, ())


def _get_skipped_existing_singleton():
    return SKIPPED_EXISTING


SKIPPED_EXISTING = _SkipExisting()


SUPPORTED_CALCULATOR_NAMES = {
    "mace",
    "mace-polar",
    "xtb",
    "emt",
    "orca",
    "exachem",
    "uma",
    "uma-s-omol",
    "uma-s-omat",
    "uma-s-odac",
    "uma-m-omol",
    "uma-m-omat",
    "uma-m-odac",
}


def _resolve_role_calculators(run_params, roles, calc_params):
    """Instantiate per-role calculator overrides from YAML task params.

    Mutates ``run_params`` by popping recognised role keys. Raises RuntimeError
    on any failure so the caller (MPI rank or Parsl @python_app) can decide
    whether to abort the world or fail just this row.
    """

    from iqc.asetools import get_calculator

    role_calculators = {}
    for role in roles:
        name = run_params.pop(role, None)
        if name is None:
            continue
        if isinstance(name, str):
            if name.lower() not in SUPPORTED_CALCULATOR_NAMES:
                raise RuntimeError(
                    f"Unknown calculator '{name}' for {role}. "
                    f"Supported names: {sorted(SUPPORTED_CALCULATOR_NAMES)}. "
                    "To use a calculator outside this list, pass an "
                    "instantiated calculator via the Python API."
                )
            try:
                instance = get_calculator(name=name, **calc_params)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to initialize {role} '{name}': {e}"
                ) from e
            # get_calculator silently falls back to MACE when its target fails.
            # For per-role overrides the user explicitly asked for a calculator,
            # so fail instead of quietly changing methods.
            fallback_from = getattr(instance, "_iqc_fallback_from", None)
            if fallback_from is not None:
                raise RuntimeError(
                    f"Requested {role}='{name}' but get_calculator silently "
                    f"fell back to MACE (was: '{fallback_from}'). Check earlier "
                    "warnings — typical causes: missing executable (e.g. ORCA "
                    "not on PATH and ASE_ORCA_COMMAND unset), missing Python "
                    "package, or initialization failure."
                )
            role_calculators[role] = instance
        else:
            role_calculators[role] = name
    return role_calculators


def _process_one_row(
    xyz_index,
    *,
    args,
    params_str,
    task,
    calculator_name,
    calc_params,
    opt_params,
    vib_params,
    ir_params,
    thermo_params,
    nmr_params,
    xyz_files,
    input_mode,
    number_of_files,
    calculator,
    worker_id,
    n_workers,
    rank_output_dir_factory,
    direct_work_dir,
    completed_file_index,
    db_path,
):
    """Process a single input row.

    Three possible returns:

    - **dict** — task ran (possibly with ``{task}_error`` set on failure).
      Includes helper keys ``_unique_name``, ``_record_stamp``,
      ``_work_dir_used`` for the caller to consume and strip.
    - **None** — input could not be read (bad xyz / unknown error). Caller
      should not persist anything and should not bump skip counters.
    - **SKIPPED_EXISTING** sentinel — skip-existing matched. Caller should
      not persist anything but should increment its ``skipped_existing`` count.

    The caller is responsible for persisting the returned dict (per-rank JSON
    file in MPI mode; appended to a shared JSONL in the Parsl head).
    """

    from iqc.asetools import (
        atoms2xyz,
        get_ase_version,
        get_atoms_from_smiles,
        get_atoms_from_xyz,
        run_ir,
        run_ir_thermo,
        run_optimization,
        run_single_point,
        run_thermo,
        run_vibrations,
    )
    from iqc.nmr import run_nmr_workflow

    # --- Resolve input descriptor -------------------------------------------
    smiles_input = None
    xyz_record = None
    smiles_record = None
    if input_mode == "smiles":
        smiles_input = xyz_files[0]
        xyz_file = f"smiles:{smiles_input}"
        base_name = _smiles_to_basename(smiles_input)
    elif input_mode == "data_xyz":
        xyz_record = xyz_files[xyz_index]
        xyz_file = f"{args.input}:{args.xyz}[{xyz_record.row_index}]"
        base_name = f"{Path(args.input).stem}_row{xyz_record.row_index}"
    elif input_mode == "data_smiles":
        smiles_record = xyz_files[xyz_index]
        smiles_input = smiles_record.smiles
        xyz_file = f"{args.input}:{args.smiles}[{smiles_record.row_index}]"
        base_name = f"{Path(args.input).stem}_row{smiles_record.row_index}"
    elif number_of_files > 1:
        xyz_file = xyz_files[xyz_index]
        base_name = os.path.splitext(os.path.basename(xyz_file))[0]
    else:
        xyz_file = xyz_files[0]
        base_name = os.path.splitext(os.path.basename(xyz_file))[0]
    record_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{base_name}_{xyz_index}_{worker_id}_{record_stamp}"
    logging.info(f"Processing input: {xyz_file} with unique ID: {unique_name}")

    # --- Read atoms ----------------------------------------------------------
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
    except ValueError as e:
        logging.error(f"Error reading file {xyz_file}: {e}. Skipping.")
        return None
    except Exception as e:
        logging.error(
            f"Unexpected error processing file {xyz_file}: {e}. Skipping."
        )
        return None

    # --- Initial result metadata --------------------------------------------
    # NOTE: ExaChem-specific energy components and method metadata
    # (scf_energy_eV, mp2_correlation_eV, ccsd_correlation_eV,
    # t_correction_eV, total_energy_eV, scf_time_s, ccsd_time_s, t_time_s,
    # basis, scf_type, method, frozen_core) are merged into this dict at the
    # ``results.update(task_results)`` call below — they live at the top level
    # and are queryable directly from JSONL/SQLite, not buried inside
    # ``exachem_output``. See ``ExaChemCalculator._extract_components``.
    model_name = getattr(calculator, "model_name", "") if calculator else ""
    results = {
        "xyz_file": xyz_file,
        "smiles_input": smiles_input or "",
        "input_mode": input_mode,
        "data_input_file": args.input or "",
        "data_xyz_column": args.xyz if input_mode == "data_xyz" else "",
        "data_smiles_column": args.smiles if input_mode == "data_smiles" else "",
        "data_sort_column": args.sort or "",
        "data_sort_order": args.sort_order if args.sort else "",
        "data_row_index": (
            xyz_record.row_index
            if input_mode == "data_xyz"
            else smiles_record.row_index if input_mode == "data_smiles" else ""
        ),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mpi_size": n_workers,
        "mpi_rank": worker_id,
        "hostname": os.uname().nodename,
        "ase_version": get_ase_version(),
        "task": task,
        "calculator": calculator_name,
        "model": model_name,
        "initial_xyz": atoms2xyz(atoms),
        "params": params_str,
    }

    # --- Skip-existing -------------------------------------------------------
    if args.skip_existing:
        current_key = calculation_key(
            results["initial_xyz"],
            results["params"],
            results["calculator"],
            results["model"],
            results["task"],
        )
        # With --retry-failed-only, the index already excludes error rows so a
        # plain `in` check below preserves the F5 contract: skip only successes.
        require_ok = getattr(args, "retry_failed_only", False)
        skip_source = None
        index_status = completed_file_index.get(current_key) if isinstance(
            completed_file_index, dict
        ) else ("ok" if current_key in completed_file_index else None)
        if index_status is not None and (not require_ok or index_status == "ok"):
            skip_source = "result files"
        elif db_path and calculation_exists(
            db_path,
            results["initial_xyz"],
            results["params"],
            results["calculator"],
            results["model"],
            results["task"],
            include_errors=not require_ok,
        ):
            skip_source = "database"

        if skip_source:
            logging.info(
                "Skipping existing %s calculation for input %s; found in %s.",
                task,
                xyz_file,
                skip_source,
            )
            return SKIPPED_EXISTING

    # --- Lazy work-dir helper (mirrors the closure in main()) ---------------
    work_dir_used = [False]

    def _get_work_dir():
        work_dir_used[0] = True
        if args.direct_db:
            return direct_work_dir
        return rank_output_dir_factory()

    # --- Resolve charge / multiplicity --------------------------------------
    ase_multiplicity = (
        args.multiplicity
        if getattr(args, "multiplicity", None) is not None
        else atoms.info.get("multiplicity")
    )
    ase_charge = (
        args.charge
        if getattr(args, "charge", None) is not None
        else int(atoms.info.get("charge", 0))
    )

    # --- Dispatch task -------------------------------------------------------
    try:
        if task == "single":
            atoms, task_results = run_single_point(
                atoms=atoms,
                calculator=calculator,
                unique_name=unique_name,
                multiplicity=ase_multiplicity,
                charge=ase_charge,
            )
        elif task == "opt":
            opt_run_params = dict(opt_params)
            trajectory_file = None
            save_geometry = False
            if args.save:
                output_dir = _get_work_dir()
                trajectory_file = os.path.join(
                    output_dir, f"{unique_name}_opt_trajectory.traj"
                )
                opt_run_params["output_dir"] = output_dir
                save_geometry = True
            explicit_params = {
                "atoms",
                "calculator",
                "unique_name",
                "trajectory",
                "save_geometry",
            }
            opt_params_filtered = {
                k: v for k, v in opt_run_params.items() if k not in explicit_params
            }
            atoms, task_results = run_optimization(
                atoms=atoms,
                calculator=calculator,
                unique_name=unique_name,
                trajectory=trajectory_file,
                save_geometry=save_geometry,
                multiplicity=ase_multiplicity,
                charge=ase_charge,
                **opt_params_filtered,
            )
        elif task == "vib":
            vib_run_params = dict(vib_params)
            output_dir = _get_work_dir()
            vib_run_params["vib_dir"] = output_dir
            trajectory_file = None
            save_geometry = False
            if args.save:
                trajectory_file = os.path.join(
                    output_dir, f"{unique_name}_vib_trajectory.traj"
                )
                vib_run_params["output_dir"] = output_dir
                save_geometry = True
            explicit_params = {
                "atoms",
                "calculator",
                "optimize",
                "unique_name",
                "trajectory",
                "save_geometry",
            }
            vib_params_filtered = {
                k: v for k, v in vib_run_params.items() if k not in explicit_params
            }
            atoms, task_results = run_vibrations(
                atoms=atoms,
                calculator=calculator,
                optimize=True,
                unique_name=unique_name,
                trajectory=trajectory_file,
                save_geometry=save_geometry,
                multiplicity=ase_multiplicity,
                charge=ase_charge,
                **vib_params_filtered,
            )
        elif task == "ir":
            ir_run_params = dict(ir_params)
            output_dir = _get_work_dir()
            ir_run_params["vib_dir"] = output_dir
            trajectory_file = None
            save_geometry = False
            if args.save:
                trajectory_file = os.path.join(
                    output_dir, f"{unique_name}_ir_trajectory.traj"
                )
                ir_run_params["output_dir"] = output_dir
                save_geometry = True

            role_calculators = _resolve_role_calculators(
                ir_run_params,
                (
                    "optimization_calculator",
                    "vibration_calculator",
                    "dipole_calculator",
                ),
                calc_params,
            )

            explicit_params = {
                "atoms",
                "calculator",
                "optimization_calculator",
                "vibration_calculator",
                "dipole_calculator",
                "optimize",
                "unique_name",
                "trajectory",
                "save_geometry",
            }
            ir_params_filtered = {
                k: v for k, v in ir_run_params.items() if k not in explicit_params
            }
            atoms, task_results = run_ir(
                atoms=atoms,
                calculator=calculator,
                optimize=True,
                unique_name=unique_name,
                trajectory=trajectory_file,
                save_geometry=save_geometry,
                multiplicity=ase_multiplicity,
                charge=ase_charge,
                **role_calculators,
                **ir_params_filtered,
            )
        elif task == "ir-thermo":
            ignore_imag = args.ignore_imag
            ir_thermo_run_params = {**thermo_params, **ir_params}
            output_dir = _get_work_dir()
            ir_thermo_run_params["vib_dir"] = output_dir
            trajectory_file = None
            save_geometry = False
            if args.save:
                trajectory_file = os.path.join(
                    output_dir, f"{unique_name}_ir-thermo_trajectory.traj"
                )
                ir_thermo_run_params["output_dir"] = output_dir
                save_geometry = True

            role_calculators = _resolve_role_calculators(
                ir_thermo_run_params,
                (
                    "optimization_calculator",
                    "vibration_calculator",
                    "dipole_calculator",
                ),
                calc_params,
            )

            explicit_params = {
                "atoms",
                "calculator",
                "optimization_calculator",
                "vibration_calculator",
                "dipole_calculator",
                "optimize",
                "ignore_imag_modes",
                "unique_name",
                "trajectory",
                "save_geometry",
            }
            ir_thermo_params_filtered = {
                k: v
                for k, v in ir_thermo_run_params.items()
                if k not in explicit_params
            }
            atoms, task_results = run_ir_thermo(
                atoms=atoms,
                calculator=calculator,
                optimize=True,
                ignore_imag_modes=ignore_imag,
                unique_name=unique_name,
                trajectory=trajectory_file,
                save_geometry=save_geometry,
                multiplicity=ase_multiplicity,
                charge=ase_charge,
                **role_calculators,
                **ir_thermo_params_filtered,
            )
        elif task == "thermo":
            ignore_imag = args.ignore_imag
            thermo_run_params = dict(thermo_params)
            output_dir = _get_work_dir()
            thermo_run_params["vib_dir"] = output_dir
            trajectory_file = None
            save_geometry = False
            if args.save:
                trajectory_file = os.path.join(
                    output_dir, f"{unique_name}_thermo_trajectory.traj"
                )
                thermo_run_params["output_dir"] = output_dir
                save_geometry = True
            explicit_params = {
                "atoms",
                "calculator",
                "unique_name",
                "ignore_imag_modes",
                "trajectory",
                "save_geometry",
            }
            thermo_params_filtered = {
                k: v
                for k, v in thermo_run_params.items()
                if k not in explicit_params
            }
            atoms, task_results = run_thermo(
                atoms=atoms,
                calculator=calculator,
                unique_name=unique_name,
                ignore_imag_modes=ignore_imag,
                trajectory=trajectory_file,
                save_geometry=save_geometry,
                multiplicity=ase_multiplicity,
                charge=ase_charge,
                **thermo_params_filtered,
            )
        elif task == "nmr":
            nmr_run_params = {**nmr_params, **get_nmr_cli_overrides(args)}
            if not nmr_run_params.get("output_dir"):
                nmr_run_params["output_dir"] = os.path.join(
                    _get_work_dir(), f"{unique_name}_nmr"
                )
            atoms, task_results = run_nmr_workflow(
                atoms=atoms,
                unique_name=unique_name,
                **nmr_run_params,
            )
        else:
            raise ValueError(f"Unsupported task '{task}'")

        results.update(task_results)
        logging.debug(f"Completed {task} calculations for file: {xyz_file}")
    except Exception as e:
        results[f"{task}_error"] = str(e)
        logging.error(f"Task '{task}' failed for {xyz_file}: {e}", exc_info=True)

    results["_unique_name"] = unique_name
    results["_record_stamp"] = record_stamp
    results["_work_dir_used"] = work_dir_used[0]
    return results


def main():
    """Main function."""
    start_time = time.time()

    # Get command line arguments
    args = get_args()
    if args.input and args.input_only:
        return run_input_inspection(args.input)
    input_error = validate_input_args(args)
    if input_error:
        print(input_error, file=sys.stderr)
        return 1

    # ASE must be imported before MPI initialization for calculation workflows.
    import ase  # noqa: F401
    import ase.parallel as asepar

    asepar.world = asepar.DummyMPI()

    from iqc.asetools import (
        MACE_POLAR_DEFAULT_MODEL,
        _ensure_mace_polar_model_cached,
        atoms2xyz,
        get_ase_version,
        get_atoms_from_smiles,
        get_atoms_from_xyz,
        get_calculator,
        run_ir,
        run_ir_thermo,
        run_optimization,
        run_single_point,
        run_thermo,
        run_vibrations,
    )
    from iqc.mpitools import get_mpi_context, get_start_end
    from iqc.nmr import run_nmr_workflow
    from iqc.xyztools import count_xyz_frames

    comm, mpi = get_mpi_context()
    rank = comm.Get_rank()
    size = comm.Get_size()
    task = args.task
    run_id = comm.bcast(_make_run_id() if rank == 0 else None, root=0)
    direct_work_dir = comm.bcast(
        (
            _unique_child_path(args.scratch, f"iqc_{task}_{run_id}")
            if rank == 0
            else None
        ),
        root=0,
    )

    # --- Logging Setup --- (Remains mostly the same)
    logger = logging.getLogger()
    log_level_name = args.loglevel.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "IQC %(levelname)s: %(asctime)s - Rank %(mpi_rank)s - %(message)s"
    )
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(formatter)
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.mpi_rank = rank
        return record

    logging.setLogRecordFactory(record_factory)
    logging.debug(f"Number of MPI ranks: {size}.")
    # --- Load Parameters from File ---
    params = {}
    params_str = ""
    if args.params and os.path.isfile(args.params):
        try:
            with open(args.params, "r") as f:
                params_str = f.read()
                params = yaml.safe_load(params_str) or {}
            if rank == 0:
                logging.info(f"Loaded parameters from {args.params}")
                logging.debug(f"Parameters: {params}")

        except Exception as e:
            logging.error(f"Error loading parameters from {args.params}: {e}")
            # Decide if execution should stop if params file is bad.
            # Use comm.Abort under MPI so peer ranks aren't left waiting.
            if size > 1:
                comm.Abort(1)
            sys.exit(1)

    elif args.params:
        if rank == 0:
            logging.warning(
                f"Parameter file specified ({args.params}) but not found. Using defaults."
            )
    # Extract specific parameter sections, defaulting to empty dicts
    calc_params = params.get("calculator_params", {})
    opt_params = params.get("optimization_params", {})
    vib_params = params.get("vibration_params", {})
    ir_params = params.get("ir_params", {})
    thermo_params = params.get("thermo_params", {})
    nmr_params = params.get("nmr_params", {})
    # Merge params for cascading:
    # vib needs opt params (calls opt internally),
    # ir uses vib-compatible params + optional ir-specific overrides,
    # thermo needs vib params (calls vib which calls opt internally)
    # BUT: calc_params should NOT be merged - they're only for calculator initialization
    vib_params = {**opt_params, **vib_params}
    ir_params = {**vib_params, **ir_params}
    thermo_params = {**vib_params, **thermo_params}
    if rank == 0:
        logging.info(f"All parameters: {params}")
    calculator = None
    if task == "nmr":
        calculator_name = args.backend or nmr_params.get("backend", "orca")
    else:
        # Determine calculator name: CLI > Param file > Default ('mace')
        calculator_name = (
            args.calculator
            if args.calculator is not None
            else params.get("calculator", "mace")
        )

        if str(calculator_name).lower() == "mace-polar":
            prefetch_error = None
            if rank == 0:
                try:
                    _ensure_mace_polar_model_cached(
                        calc_params.get("model", MACE_POLAR_DEFAULT_MODEL)
                    )
                except Exception as e:
                    prefetch_error = str(e)
            prefetch_error = comm.bcast(prefetch_error, root=0)
            if prefetch_error:
                logging.error(
                    "Failed to cache MACE-Polar checkpoint before MPI startup: %s",
                    prefetch_error,
                )
                comm.Abort(1)

        # F4: route --keep-artifacts (CLI) into ExaChem calculator kwargs so
        # the manifest is emitted without requiring users to also edit their
        # YAML params file. Non-ExaChem calculators ignore this flag.
        if (
            str(calculator_name).lower() == "exachem"
            and getattr(args, "keep_artifacts", False)
        ):
            calc_params.setdefault("keep_artifacts", True)

        # F9: route --artifact-retention / --artifact-root (CLI) into the
        # ExaChem calculator's artifact_retention dict so the F9 archive
        # hook fires without needing a YAML override.
        if (
            str(calculator_name).lower() == "exachem"
            and getattr(args, "artifact_retention", "off") == "on"
        ):
            existing = calc_params.get("artifact_retention") or {}
            existing.setdefault("enabled", True)
            existing.setdefault("compress", True)
            ar = getattr(args, "artifact_root", None)
            if ar:
                existing.setdefault("destination_root", ar)
            calc_params["artifact_retention"] = existing

        # Initialize the calculator
        try:
            calculator = get_calculator(name=calculator_name, **calc_params)
        except RuntimeError as e:
            logging.error(f"Failed to initialize calculator '{calculator_name}': {e}")
            comm.Abort(1)
    input_mode = get_structure_input_mode(args)
    if rank == 0:
        if input_mode == "smiles":
            xyz_files = [args.smiles]
            number_of_xyz = 1
            number_of_files = 1
            logging.info(f"Using SMILES input: {args.smiles}")
        elif input_mode == "data_xyz":
            try:
                xyz_files = read_xyz_column_records(
                    args.input,
                    args.xyz,
                    sort_column=args.sort,
                    sort_order=args.sort_order,
                )
            except Exception as e:
                logging.error(f"Error reading XYZ column '{args.xyz}': {e}")
                comm.Abort(1)
            number_of_xyz = len(xyz_files)
            number_of_files = number_of_xyz
            logging.info(f"Using XYZ column '{args.xyz}' from data input: {args.input}")
            if args.sort:
                logging.info(
                    f"Sorted data input by column '{args.sort}' "
                    f"({args.sort_order})."
                )
            logging.info(f"Number of configurations: {number_of_xyz}")
        elif input_mode == "data_smiles":
            try:
                xyz_files = read_smiles_column_records(
                    args.input,
                    args.smiles,
                    sort_column=args.sort,
                    sort_order=args.sort_order,
                )
            except Exception as e:
                logging.error(f"Error reading SMILES column '{args.smiles}': {e}")
                comm.Abort(1)
            number_of_xyz = len(xyz_files)
            number_of_files = number_of_xyz
            logging.info(
                f"Using SMILES column '{args.smiles}' from data input: {args.input}"
            )
            if args.sort:
                logging.info(
                    f"Sorted data input by column '{args.sort}' "
                    f"({args.sort_order})."
                )
            logging.info(f"Number of configurations: {number_of_xyz}")
        elif os.path.isdir(args.xyz):
            xyz_dir = args.xyz
            xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
            number_of_xyz = len(xyz_files)
        elif os.path.isfile(args.xyz):
            xyz_files = [args.xyz]
            try:
                number_of_xyz = count_xyz_frames(args.xyz)
            except Exception as e:
                logging.error(f"Error counting .xyz files in {args.xyz}: {e}")
                comm.Abort(1)
        else:
            # Handle non-existent path before bcast
            logging.error(
                f"Input path {args.xyz} does not exist or is not a file/directory."
            )
            xyz_files = []  # Ensure empty list is broadcast
            comm.Abort(1)

        number_of_files = len(xyz_files)
        if number_of_files == 0:
            logging.error(f"No .xyz files found in {args.xyz}. Exiting.")
            comm.Abort(1)
        elif input_mode not in {"data_xyz", "data_smiles"}:
            logging.info(f"Found {number_of_files} .xyz file(s).")
            logging.info(f"Number of configurations: {number_of_xyz}")

    xyz_files = comm.bcast(xyz_files if rank == 0 else None, root=0)
    input_mode = comm.bcast(input_mode if rank == 0 else None, root=0)
    number_of_xyz = comm.bcast(number_of_xyz if rank == 0 else None, root=0)
    number_of_files = len(xyz_files)

    start_index, end_index = get_start_end(comm, number_of_xyz)
    logging.debug(f"Processing files from index {start_index} to {end_index}.")
    logging.info(f"Initialization time: {time.time() - start_time} seconds.")

    db_path = args.database
    if db_path and rank == 0:
        logging.info(f"Checking/Creating database at: {db_path}")
        create_database(db_path)
    if db_path:
        comm.Barrier()

    completed_file_index = {}
    if args.skip_existing:
        if rank == 0:
            skip_sources = (
                [Path(source).expanduser() for source in args.skip_existing_from]
                if args.skip_existing_from
                else _default_skip_existing_sources()
            )
            include_errors = not getattr(args, "retry_failed_only", False)
            completed_file_index, skip_index_summary = (
                build_completed_calculation_index(
                    skip_sources, include_errors=include_errors
                )
            )
            if db_path or completed_file_index:
                logging.info(
                    "Skip-existing enabled: indexed %s completed calculation(s) "
                    "from %s result record(s) in %s file(s).",
                    len(completed_file_index),
                    skip_index_summary["records"],
                    skip_index_summary["files"],
                )
                if db_path:
                    logging.info("Skip-existing will also check database: %s", db_path)
            else:
                logging.info(
                    "Skip-existing enabled, but no database was provided and no "
                    "existing IQC result files were found."
                )
            if skip_index_summary["invalid"]:
                logging.warning(
                    "Ignored %s invalid or non-IQC record(s) while indexing "
                    "existing result files.",
                    skip_index_summary["invalid"],
                )
        completed_file_index = comm.bcast(
            completed_file_index if rank == 0 else None, root=0
        )

    dir_name = None
    work_dir_used = False
    skipped_existing = 0

    def get_rank_output_dir():
        """Create this rank's result/scratch directory only when it is needed."""

        nonlocal dir_name
        if dir_name is None:
            dir_name = _unique_child_path(
                _rank_output_parent(), f"tmp_{task}_{rank}_{run_id}"
            )
            logging.debug(f"Creating directory: {dir_name}")
            os.makedirs(dir_name, exist_ok=True)
        return dir_name

    def get_work_dir():
        """Return the directory for files that are required by this task."""

        nonlocal work_dir_used
        work_dir_used = True
        if args.direct_db:
            return direct_work_dir
        return get_rank_output_dir()

    for xyz_index in range(start_index, end_index):
        result = _process_one_row(
            xyz_index,
            args=args,
            params_str=params_str,
            task=task,
            calculator_name=calculator_name,
            calc_params=calc_params,
            opt_params=opt_params,
            vib_params=vib_params,
            ir_params=ir_params,
            thermo_params=thermo_params,
            nmr_params=nmr_params,
            xyz_files=xyz_files,
            input_mode=input_mode,
            number_of_files=number_of_files,
            calculator=calculator,
            worker_id=rank,
            n_workers=size,
            rank_output_dir_factory=get_rank_output_dir,
            direct_work_dir=direct_work_dir,
            completed_file_index=completed_file_index,
            db_path=db_path,
        )
        if result is SKIPPED_EXISTING:
            skipped_existing += 1
            continue
        if result is None:
            # Bad input row — already logged inside _process_one_row.
            continue
        if result.pop("_work_dir_used", False):
            work_dir_used = True
        unique_name = result.pop("_unique_name")
        record_stamp = result.pop("_record_stamp")

        if args.direct_db and db_path:
            insert_entry(json.dumps(result, cls=ComplexEncoder), db_path)
        elif not args.direct_db:
            output_file = f"{unique_name}_{task}_{record_stamp}_{rank}.json"
            output_file = os.path.join(get_rank_output_dir(), output_file)
            save_results(result, output_file)

    # Wait for all processes to finish before combining files
    barrier_start = time.time()
    logging.debug(f"Waiting for all processes to finish before combining files.")
    comm.Barrier()
    logging.debug(f"Took { time.time() - barrier_start:.2f} seconds")
    total_skipped_existing = comm.reduce(skipped_existing, op=mpi.SUM, root=0)
    if args.skip_existing and rank == 0:
        logging.info("Skipped %s existing calculation(s).", total_skipped_existing)

    if not args.direct_db:
        rank_output_dirs = comm.gather(dir_name, root=0)

        # Define jsonl_file for all ranks
        jsonl_file = f"iqc_{task}_results_{run_id}.jsonl"

        if rank == 0:
            combine_start = time.time()
            logging.debug(f"Starting to combine JSON files")

            # Combine all JSON files into a single JSONL file
            json_files = []
            for output_dir in rank_output_dirs:
                if output_dir:
                    json_files.extend(
                        glob.glob(os.path.join(output_dir, f"*_{task}_*.json"))
                    )
            logging.debug(f"Found {len(json_files)} JSON files to combine.")
            with open(jsonl_file, "w") as outfile:
                for json_file in json_files:
                    try:
                        with open(json_file, "r") as infile:
                            data = json.load(infile)
                            json.dump(data, outfile)
                            outfile.write("\n")
                    except json.JSONDecodeError as e:
                        logging.warning(
                            f"Failed to parse JSON file {json_file}: {e}. Skipping."
                        )
                        continue
                    except Exception as e:
                        logging.warning(
                            f"Error reading JSON file {json_file}: {e}. Skipping."
                        )
                        continue

            combine_end = time.time()
            logging.debug(
                f"Finished combining JSON files in {combine_end - combine_start:.2f} seconds"
            )
            logging.info(f"Combined results saved to {jsonl_file}")
            try:
                convert_jsonl_results_to_parquet(jsonl_file)
            except ValueError as e:
                logging.warning(f"Skipping parquet conversion: {e}")
            except Exception as e:
                logging.error(
                    f"Failed to convert JSONL results to parquet: {e}",
                    exc_info=True,
                )
                comm.Abort(1)
            logging.info(f"Total time: {time.time() - start_time} seconds.")

        # Wait for rank 0 to finish creating the JSONL file
        comm.Barrier()

        # SQLite database - only check file existence on rank 0
        if rank == 0:
            if not os.path.exists(jsonl_file):
                logging.error(f"JSONL file not found: {jsonl_file}")
                comm.Abort(1)

        if db_path and rank == 0:

            logging.info(f"Reading from JSONL file: {jsonl_file}")
            insert_jsonl_to_db(jsonl_file, db_path)
            logging.info(f"Finished inserting data into database: {db_path}")

        elif rank == 0:
            logging.info("No database specified.")

        return 0

    else:
        any_work_dir_used = comm.allreduce(1 if work_dir_used else 0, op=mpi.MAX)
        comm.Barrier()
        if rank == 0:
            logging.info("Results were saved directly to the database.")
            if any_work_dir_used:
                logging.info(
                    f"Required scratch/output files were written under: {direct_work_dir}"
                )
            else:
                logging.info("No IQC result files or folders were created.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
