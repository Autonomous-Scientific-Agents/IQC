"""ASE calculator for ExaChem.

ExaChem (https://github.com/ExaChem/exachem) is an MPI-parallel electronic
structure code that reads a JSON input describing geometry, basis, methods,
and per-method settings, and writes JSON output containing converged energies.

This module wraps ExaChem so it can be driven through ASE conventions:
``calc.get_potential_energy()`` produces a single-point energy (in eV) for the
attached ``Atoms``. ExaChem-specific knobs (basis set, SCF type, CC
thresholds, raw input overrides) are exposed alongside the standard ASE
parameters.

The calculator does **not** compute forces — ExaChem's analytic-gradient
support is method-dependent and not exposed through this thin wrapper.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ase import units
from ase.calculators.calculator import (
    Calculator,
    CalculationFailed,
    all_changes,
)


_HARTREE_TO_EV = units.Hartree

# Default ExaChem binary location on this system. Overridable via the
# ``EXACHEM_BINARY`` environment variable or the ``binary=`` calculator
# argument.
_DEFAULT_BINARY = "/home/keceli/IQC/keceli/nwx/install_gpu/bin/ExaChem"

# Mapping from user-facing method aliases to the TASK flags ExaChem expects.
# Each value is the list of TASK keys to enable. SCF is always enabled
# implicitly by the underlying methods.
_METHOD_TASK_KEYS: Dict[str, List[str]] = {
    "scf": ["scf"],
    "hf": ["scf"],
    "mp2": ["mp2"],
    "ccsd": ["ccsd"],
    "ccsd(t)": ["ccsd_t"],
    "ccsd_t": ["ccsd_t"],
    "ccsd-t": ["ccsd_t"],
    "eom-ccsd": ["eom_ccsd"],
    "eom_ccsd": ["eom_ccsd"],
}


def _normalize_method(method: str) -> str:
    """Return the canonical ExaChem method name for a user-supplied alias."""

    key = method.lower().replace(" ", "")
    if key not in _METHOD_TASK_KEYS:
        raise ValueError(
            f"Unsupported ExaChem method '{method}'. "
            f"Choose one of: {sorted(_METHOD_TASK_KEYS)}"
        )
    return key


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` (returns ``base``)."""

    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _atoms_to_coordinate_lines(atoms) -> List[str]:
    """Format ASE atoms as ExaChem ``geometry.coordinates`` lines (angstrom)."""

    lines = []
    for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f"{symbol:<3s} {x: .12f} {y: .12f} {z: .12f}")
    return lines


# ---------------------------------------------------------------------------
# Restart-from-prior-amplitudes helpers (F10)
# ---------------------------------------------------------------------------
#
# Kinds we restage when starting from a prior calculation. Anything else
# (output JSON, logs, scratch files) is deliberately ignored — the next run
# regenerates them.
_RESTART_KINDS = ("mo", "amplitudes", "cholesky", "restart")

# Glob patterns used to classify on-disk artifacts when a F4-style manifest
# is unavailable. Same heuristics as the F4 contract so manifest-based and
# manifest-less staging produce equivalent files.
_KIND_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "mo": ("*.mo*", "*.molden"),
    "amplitudes": ("*t1*", "*t2*", "*ccsd_t*"),
    "cholesky": ("*chol*", "*cd_vec*"),
    "restart": ("*restart*", "*chkpt*"),
}


def _classify_file_kind(name: str) -> Optional[str]:
    """Return the artifact kind for a file basename, or None if not restartable."""

    for kind, patterns in _KIND_PATTERNS.items():
        for pat in patterns:
            if fnmatch.fnmatch(name, pat):
                return kind
    return None


def _looks_like_calculation_key(value: Any) -> bool:
    """True for inputs that should be resolved via the SQLite manifest index."""

    if isinstance(value, dict):
        return "geometry_hash" in value
    if isinstance(value, (list, tuple)):
        return len(value) == 5
    if isinstance(value, str):
        # 64-hex sha256 string treated as a geometry_hash digest.
        return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())
    return False


def _lookup_artifact_in_sqlite(
    restart_id: Union[str, dict, list, tuple],
    db_path: Optional[Path],
) -> Optional[Path]:
    """Look up an archived run directory in the F4 artifact_manifest blob_data.

    F4 stores run_dir / artifact_archive inside the per-row blob_data JSON
    (per the shared contract); F9 stores artifact_archive there too. We
    return the absolute path to the archive (preferred) or run_dir, or None
    when the table isn't present (F4 not merged yet) or no row matches.
    Failures are logged and swallowed so a missing table never breaks a run.
    """

    if db_path is None or not Path(db_path).exists():
        logging.warning(
            "restart_from points at a calculation key but no SQLite DB is "
            "available; cannot resolve."
        )
        return None

    try:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='calculations'"
            )
            if cursor.fetchone() is None:
                logging.warning(
                    "calculations table absent in %s; cannot resolve restart_from",
                    db_path,
                )
                return None
            rows = conn.execute("SELECT blob_data FROM calculations").fetchall()
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - defensive against schema drift
        logging.warning("Failed to query artifact manifests from %s: %s", db_path, exc)
        return None

    target_geometry_hash: Optional[str] = None
    if isinstance(restart_id, str):
        target_geometry_hash = restart_id
    elif isinstance(restart_id, dict):
        target_geometry_hash = restart_id.get("geometry_hash")
    elif isinstance(restart_id, (list, tuple)) and len(restart_id) == 5:
        target_geometry_hash = restart_id[0]

    for (blob,) in rows:
        try:
            record = json.loads(blob)
        except (TypeError, ValueError):
            continue
        if target_geometry_hash and record.get("geometry_hash") != target_geometry_hash:
            continue
        archive = record.get("artifact_archive")
        if archive and Path(archive).exists():
            return Path(archive)
        run_dir = record.get("run_dir")
        if run_dir and Path(run_dir).exists():
            return Path(run_dir)

    logging.warning(
        "No matching artifact found for restart_id %r in %s", restart_id, db_path
    )
    return None


def resolve_restart_artifact(
    restart_id: Union[str, Path, dict, list, tuple],
    *,
    db_path: Optional[Path] = None,
) -> Optional[Path]:
    """Map a restart_from value to an on-disk source (directory or .tar.gz).

    Filesystem inputs (existing paths) are returned as-is so callers without
    a SQLite manifest still work. Calculation-key dicts / hashes / tuples
    trigger an artifact_manifest lookup; missing table or no-hit returns
    None with a logged warning rather than raising.
    """

    if isinstance(restart_id, Path):
        return restart_id if restart_id.exists() else None

    if isinstance(restart_id, str):
        candidate = Path(restart_id)
        if candidate.exists():
            return candidate
        # Not an on-disk path -- fall through to manifest lookup below.

    if _looks_like_calculation_key(restart_id):
        return _lookup_artifact_in_sqlite(restart_id, db_path)

    return None


def _stage_restart_artifacts(
    source: Path,
    run_dir: Path,
    *,
    manifest: Optional[List[Dict[str, Any]]] = None,
) -> List[Path]:
    """Copy restart-relevant files from ``source`` into ``run_dir``.

    ``source`` is either a directory (typically a prior ExaChem run_dir) or
    a ``.tar.gz`` archive produced by F9. When ``manifest`` is supplied
    (F4-shaped list of dicts with ``path`` + ``kind``), it drives selection;
    otherwise files matching the kind patterns are picked up by basename.
    Returns the list of staged destination paths.
    """

    if not source.exists():
        raise CalculationFailed(f"restart_from source does not exist: {source}")

    run_dir.mkdir(parents=True, exist_ok=True)

    extracted: Optional[tempfile.TemporaryDirectory] = None
    try:
        suffixes = "".join(source.suffixes[-2:]) if source.is_file() else ""
        if source.is_file() and (suffixes in (".tar.gz",) or source.suffix == ".tgz"):
            extracted = tempfile.TemporaryDirectory(prefix="exachem_restart_")
            with tarfile.open(source, "r:gz") as tf:
                tf.extractall(extracted.name)
            walk_root = Path(extracted.name)
        elif source.is_dir():
            walk_root = source
        else:
            raise CalculationFailed(
                f"restart_from source must be a directory or .tar.gz: {source}"
            )

        staged: List[Path] = []

        if manifest:
            # Manifest-driven: trust the kinds F4 already classified.
            for entry in manifest:
                kind = entry.get("kind")
                src_path_str = entry.get("path")
                if kind not in _RESTART_KINDS or not src_path_str:
                    continue
                src_path = Path(src_path_str)
                if extracted is not None and src_path.is_absolute():
                    matches = list(walk_root.rglob(src_path.name))
                    if not matches:
                        continue
                    src_path = matches[0]
                if not src_path.exists():
                    continue
                dest = run_dir / src_path.name
                shutil.copy2(src_path, dest)
                staged.append(dest)
        else:
            # Glob-pattern fallback (no manifest available).
            seen: set = set()
            for path in walk_root.rglob("*"):
                if not path.is_file():
                    continue
                if _classify_file_kind(path.name) is None:
                    continue
                if path.name in seen:
                    continue
                seen.add(path.name)
                dest = run_dir / path.name
                shutil.copy2(path, dest)
                staged.append(dest)

        return staged
    finally:
        if extracted is not None:
            extracted.cleanup()


class ExaChemCalculator(Calculator):
    """Drive ExaChem from ASE for single-point energy calculations.

    Standard ASE parameters:
        charge (int): Molecular charge. Default 0.
        multiplicity (int): Spin multiplicity 2S+1. Defaults to 1 for
            even-electron, 2 for odd-electron systems.

    ExaChem-specific parameters:
        method (str): 'scf'/'hf', 'mp2', 'ccsd', 'ccsd(t)' (alias 'ccsd_t'),
            'eom-ccsd'. Default 'scf'.
        basis (str): Basis set name. Default 'cc-pvdz'.
        scf_type (str): 'restricted' / 'unrestricted' / 'rohf'. Defaults to
            'restricted' for closed-shell singlets, 'unrestricted' otherwise.
        nproc (int): Number of MPI ranks. Default 1.
        mpi_command (str|list): MPI launcher. Default 'mpiexec'. Pass a list
            to inject extra flags, e.g. ``['mpiexec', '--bind-to', 'core']``.
        omp_num_threads (int|None): Sets ``OMP_NUM_THREADS`` for the run.
            Default 1. Pass ``None`` to inherit the parent environment.
        binary (str): Path to the ExaChem executable. Defaults to
            ``$EXACHEM_BINARY`` or the known local install.
        scf, cc, cd, common, basis_block, task, dplot, gw, fci (dict):
            ExaChem JSON sections merged on top of the defaults. ``basis_block``
            is renamed to ``basis`` in the final JSON to avoid clashing with
            the top-level ``basis`` string parameter.
        exachem_input (dict): Free-form dict deep-merged into the final input
            JSON last, after all other parameter-derived sections. Use this for
            anything not covered by the named sections above.
        keep_files (bool): If False (default), the per-call scratch directory
            is removed on the next call. Set True to retain all run artifacts.
        keep_artifacts (bool): If True, build an ``artifact_manifest`` listing
            MO/amplitude/cholesky/restart/output files in the run directory
            and surface the absolute ``run_dir`` and per-file
            sha256/size/kind through ``self.results`` so the orchestrator can
            persist them. Implies ``keep_files=True`` for the duration of
            this run so the listed files are not wiped on the next call.
            Default False (backward compatible — empty manifest, null run_dir).
    """

    implemented_properties = ["energy"]

    default_parameters: Dict[str, Any] = {
        "method": "scf",
        "basis": "cc-pvdz",
        "charge": 0,
        "multiplicity": None,
        "scf_type": None,
        "nproc": 1,
        "mpi_command": "mpiexec",
        "omp_num_threads": 1,
        "binary": None,
        "scf": None,
        "cc": None,
        "cd": None,
        "common": None,
        "basis_block": None,
        "task": None,
        "dplot": None,
        "gw": None,
        "fci": None,
        "exachem_input": None,
        "keep_files": False,
        "keep_artifacts": False,
        # F9: artifact retention config. Disabled by default at the calculator
        # level — the orchestrator / CLI enables it for production sweeps so
        # interactive single-shot calls don't accumulate archives.
        "artifact_retention": None,
        "el_nnodes": None,
        "el_ppn": None,
    }

    def __init__(
        self,
        restart: Optional[str] = None,
        ignore_bad_restart_file: bool = Calculator._deprecated,
        label: str = "exachem",
        atoms=None,
        directory: str = ".",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            restart=restart,
            ignore_bad_restart_file=ignore_bad_restart_file,
            label=label,
            atoms=atoms,
            directory=directory,
            **kwargs,
        )
        # Provide a stable handle for code paths in asetools that detect
        # calculator families via `_iqc_calculator_family`.
        self._iqc_calculator_family = "exachem"
        self._iqc_spin_charge_convention = "exachem"
        # Tracks output JSON of most recent run for inspection/tests.
        self.last_output_path: Optional[Path] = None
        self.last_input_path: Optional[Path] = None
        self.last_stdout_path: Optional[Path] = None
        self.last_run_dir: Optional[Path] = None
        self.last_output_payload: Optional[Dict[str, Any]] = None
        self.cluster_client = None

    # ------------------------------------------------------------------
    # Public ASE entry point
    # ------------------------------------------------------------------
    def calculate(
        self,
        atoms=None,
        properties: Sequence[str] = ("energy",),
        system_changes: Iterable[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.atoms is None:
            raise CalculationFailed("ExaChemCalculator: no atoms attached")

        params = self.parameters
        method = _normalize_method(params["method"])
        keep_artifacts = bool(params.get("keep_artifacts", False))
        # keep_artifacts implies keep_files for the lifetime of this run so
        # the artifact files we hash/manifest aren't wiped on the next call.
        keep_files = bool(params.get("keep_files", False)) or keep_artifacts
        run_dir = self._prepare_run_dir(keep_files)
        input_path = run_dir / "input.json"
        stdout_path = run_dir / "exachem.log"

        # Stage prior-run artifacts (MO files, T1/T2 amplitudes, Cholesky
        # vectors, restart blobs) into the new run_dir before launching
        # ExaChem. The restart_from value is consumed here and stripped
        # from the JSON input by _build_input_json.
        restart_from = None
        exachem_input = params.get("exachem_input")
        if isinstance(exachem_input, dict) and "restart_from" in exachem_input:
            restart_from = exachem_input["restart_from"]
        if restart_from is not None:
            self._stage_restart_inputs(restart_from, run_dir)

        input_json = self._build_input_json(self.atoms, params, method)
        with open(input_path, "w") as fh:
            json.dump(input_json, fh, indent=2)

        cmd = self._build_command(params, input_path)
        env = self._build_env(params)

        logging.info(
            "Running ExaChem: method=%s basis=%s nproc=%s in %s",
            method,
            params["basis"],
            params["nproc"],
            run_dir,
        )

        if self.cluster_client is not None:
            self._run_via_cluster(cmd, env, run_dir, stdout_path, params)
        else:
            with open(stdout_path, "w") as log_fh:
                completed = subprocess.run(
                    cmd,
                    cwd=str(run_dir),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )
            if completed.returncode != 0:
                tail = self._tail(stdout_path, 40)
                raise CalculationFailed(
                    f"ExaChem exited with code {completed.returncode}. "
                    f"Command: {' '.join(cmd)}\nLog tail:\n{tail}"
                )

        self.last_input_path = input_path
        self.last_stdout_path = stdout_path
        self.last_run_dir = run_dir

        output_path, payload = self._locate_output(
            run_dir, input_path.stem, input_json
        )
        self.last_output_path = output_path
        self.last_output_payload = payload

        energy_hartree = self._extract_energy(payload, method)
        self.results = {"energy": energy_hartree * _HARTREE_TO_EV}
        # Expose the raw payload for downstream tooling that wants the SCF
        # breakdown, iteration counts, etc.
        self.results["exachem_output"] = payload
        self.results["energy_hartree"] = energy_hartree

        # Top-level structured fields (energies in eV, timings in seconds,
        # method/basis metadata as strings) so the orchestrator can write them
        # directly to JSONL/SQLite without poking into ``exachem_output``.
        components = self._extract_components(payload, method)
        self.results.update(components)

        # Artifact manifest (run_dir + per-file sha256/size/kind). Only built
        # when keep_artifacts is True so the default behaviour stays cheap and
        # backward compatible; F9's archive stage consumes this manifest.
        if keep_artifacts:
            manifest = self._build_artifact_manifest(run_dir, output_path)
            self.results["run_dir"] = str(run_dir.resolve())
            self.results["artifact_manifest"] = manifest
            self.results["keep_artifacts"] = True
        else:
            self.results["run_dir"] = None
            self.results["artifact_manifest"] = []
            self.results["keep_artifacts"] = False
        # F9 populates this; F4 leaves it null so downstream code can rely on
        # the field being present even when the archive stage is disabled.
        self.results["artifact_archive"] = None

        # F9: archive run_dir before any cleanup so MOs/amplitudes survive
        # scratch wipes. Consumes the manifest + run_dir populated above; if
        # keep_artifacts is False those are empty/None and the archive helper
        # quietly skips.
        self._maybe_archive_artifacts(params)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _maybe_archive_artifacts(self, params: Dict[str, Any]) -> None:
        """Archive ExaChem run_dir into the configured artifact root, if enabled."""

        retention = params.get("artifact_retention") or {}
        if not retention.get("enabled", False):
            return

        # Until F4 lands, the manifest is not built — quietly skip rather than
        # creating useless 1-file archives that don't preserve MO/amplitude
        # provenance.
        manifest = self.results.get("artifact_manifest")
        run_dir = self.results.get("run_dir") or self.results.get("run_dir_path")
        if not manifest or not run_dir:
            return

        # Import lazily so users without artifact retention configured don't
        # pay the import cost on every calculate().
        from iqc.artifact_manager import archive_run_dir  # noqa: WPS433

        destination_root = retention.get("destination_root")
        compress = bool(retention.get("compress", True))
        record = archive_run_dir(
            Path(run_dir),
            manifest,
            destination_root=Path(destination_root) if destination_root else None,
            compress=compress,
        )
        self.results["artifact_archive"] = record["archive_path"]
        self.results["artifact_archive_sha256"] = record["archive_sha256"]
        self.results["artifact_archive_size_bytes"] = record["archive_size_bytes"]


    def _prepare_run_dir(self, keep_files: bool) -> Path:
        """Return a fresh unique run directory under ``self.directory``.

        Concurrent calculate() calls (e.g. one ExaChemCalculator instance
        reused across many Parsl/EL tasks per worker) MUST get disjoint
        run_dirs or they overwrite each other's input.json / output and
        ExaChem fails with "Could not locate ExaChem JSON output".
        Each call therefore mkdtemp's its own sibling under the base.

        When ``keep_files`` is False, this instance's previous run_dir is
        rm-tree'd first so long-lived workers processing many rows do not
        accumulate hundreds of stale exachem_run_* siblings. Concurrent
        instances each track their own ``last_run_dir`` so there is no
        cross-instance interference.
        """

        base = Path(self.directory).resolve()
        base.mkdir(parents=True, exist_ok=True)
        if not keep_files and self.last_run_dir is not None:
            try:
                shutil.rmtree(self.last_run_dir)
            except (FileNotFoundError, OSError):
                pass
        return Path(tempfile.mkdtemp(prefix="exachem_run_", dir=base))

    def _stage_restart_inputs(
        self, restart_from: Any, run_dir: Path
    ) -> List[Path]:
        """Resolve and copy restart artifacts into ``run_dir``.

        Accepts a filesystem path / Path / .tar.gz, a calculation_key dict,
        a 5-tuple key, or a 64-hex geometry hash. Returns the staged
        destinations (empty if nothing was resolved); logs a warning rather
        than failing when a hash-based lookup misses (F4 may not yet be
        merged in production).
        """

        # Allow callers to pass a dict like
        # {"source": <path>, "manifest": [...], "db_path": <path>}
        # to provide an explicit manifest alongside the source.
        manifest: Optional[List[Dict[str, Any]]] = None
        db_path: Optional[Path] = None
        if isinstance(restart_from, dict) and "source" in restart_from:
            source_value = restart_from["source"]
            manifest = restart_from.get("manifest")
            if restart_from.get("db_path") is not None:
                db_path = Path(restart_from["db_path"])
        else:
            source_value = restart_from
            env_db = os.environ.get("IQC_SQLITE_DB")
            db_path = Path(env_db) if env_db else None

        source = resolve_restart_artifact(source_value, db_path=db_path)
        if source is None:
            logging.warning(
                "Could not resolve restart_from=%r; proceeding without "
                "restart artifacts.",
                restart_from,
            )
            return []

        staged = _stage_restart_artifacts(source, run_dir, manifest=manifest)
        logging.info(
            "Staged %d restart artifact(s) from %s into %s",
            len(staged),
            source,
            run_dir,
        )
        return staged

    @staticmethod
    def _resolve_binary(params: Dict[str, Any]) -> str:
        binary = (
            params.get("binary")
            or os.environ.get("EXACHEM_BINARY")
            or _DEFAULT_BINARY
        )
        if not Path(binary).is_file():
            located = shutil.which(binary) or shutil.which("ExaChem")
            if located:
                return located
            raise CalculationFailed(
                f"ExaChem binary not found at '{binary}'. Set the binary= "
                "parameter or EXACHEM_BINARY env var."
            )
        return binary

    @classmethod
    def _build_command(cls, params: Dict[str, Any], input_path: Path) -> List[str]:
        binary = cls._resolve_binary(params)
        nproc = int(params.get("nproc", 1))
        if nproc < 1:
            raise ValueError(f"nproc must be >= 1 (got {nproc})")

        mpi_command = params.get("mpi_command") or "mpiexec"
        if isinstance(mpi_command, str):
            launcher: List[str] = [mpi_command]
        else:
            launcher = list(mpi_command)

        # Only inject -n when the user hasn't already provided one.
        if not any(flag in launcher for flag in ("-n", "-np", "--n", "--np")):
            launcher.extend(["-n", str(nproc)])

        return [*launcher, binary, str(input_path)]

    @staticmethod
    def _build_env(params: Dict[str, Any]) -> Dict[str, str]:
        env = os.environ.copy()
        omp = params.get("omp_num_threads")
        if omp is not None:
            env["OMP_NUM_THREADS"] = str(int(omp))
        return env

    def _run_via_cluster(
        self,
        cmd: List[str],
        env: Dict[str, str],
        run_dir: Path,
        stdout_path: Path,
        params: Dict[str, Any],
    ) -> None:
        from ensemble_launcher.ensemble import Task
        import uuid as _uuid

        el_nnodes = params.get("el_nnodes") or 1
        el_ppn = params.get("el_ppn") or params.get("nproc", 1)

        binary = self._resolve_binary(params)
        bin_idx = cmd.index(binary)
        bare_cmd = " ".join(cmd[bin_idx:])

        # ngpus_per_process=1: each of the ppn ranks per node claims one GPU
        # slot. Combined with SystemConfig(ngpus=13, gpus=[0..11,0]) on the
        # dispatcher side, the scheduler treats ExaChem as a full-node
        # consumer and refuses to co-schedule a second instance on the same
        # node — preventing the multi-instance tile contention that trips
        # TAMM OOM (run 8646411) and CH4 CCSD-iterations stalls (run 8648167).
        # EL also uses this to export ZE_AFFINITY_MASK=<gpus[rank]> per rank
        # via async_mpi_executor's affinity script (gen_affinity_bash_script*).
        task = Task(
            task_id=f"exachem-{_uuid.uuid4().hex[:8]}",
            nnodes=el_nnodes,
            ppn=el_ppn,
            ngpus_per_process=1,
            executable=bare_cmd,
            executor_name="async_mpi",
            env={k: v for k, v in env.items()
                 if k not in os.environ or os.environ[k] != v},
            stdout_file=str(stdout_path),
            run_dir=str(run_dir),
        )
        future = self.cluster_client.submit(task)
        try:
            future.result()
        except Exception:
            tail = self._tail(stdout_path, 40) if stdout_path.exists() else ""
            raise CalculationFailed(
                f"ExaChem cluster task failed. "
                f"Command: {bare_cmd}\nLog tail:\n{tail}"
            )

    @staticmethod
    def _default_scf_type(multiplicity: int) -> str:
        return "restricted" if multiplicity == 1 else "unrestricted"

    @classmethod
    def _build_input_json(
        cls,
        atoms,
        params: Dict[str, Any],
        method: str,
    ) -> Dict[str, Any]:
        n_electrons = sum(atoms.get_atomic_numbers()) - int(params.get("charge", 0))
        mult = params.get("multiplicity")
        if mult is None:
            mult = 1 if n_electrons % 2 == 0 else 2
        mult = int(mult)
        scf_type = params.get("scf_type") or cls._default_scf_type(mult)

        # ExaChem rejects inputs that enable more than one TASK at a time.
        # Higher-level methods imply the SCF step internally.
        task_section: Dict[str, Any] = {}
        for key in _METHOD_TASK_KEYS[method]:
            task_section[key] = True

        scf_section: Dict[str, Any] = {
            "charge": int(params.get("charge", 0)),
            "multiplicity": mult,
            "scf_type": scf_type,
            "conve": 1e-8,
            "convd": 1e-7,
            "diis_hist": 10,
            "tol_lindep": 1e-6,
            "writem": 10,
        }

        cc_section: Dict[str, Any] = {
            "threshold": 1e-6,
            "ndiis": 5,
            "ccsd_maxiter": 100,
        }

        cd_section: Dict[str, Any] = {
            "diagtol": 1e-12,
            "max_cvecs": 40,
        }

        input_json: Dict[str, Any] = {
            "geometry": {
                "coordinates": _atoms_to_coordinate_lines(atoms),
                "units": "angstrom",
            },
            "basis": {"basisset": str(params.get("basis", "cc-pvdz"))},
            "common": {"maxiter": 100},
            "SCF": scf_section,
            "CD": cd_section,
            "CC": cc_section,
            "TASK": task_section,
        }

        # Merge user-provided per-section overrides (order matters: named
        # sections first, then the catch-all exachem_input).
        section_overrides: List[Tuple[str, Any]] = [
            ("common", params.get("common")),
            ("SCF", params.get("scf")),
            ("CC", params.get("cc")),
            ("CD", params.get("cd")),
            ("TASK", params.get("task")),
            ("DPLOT", params.get("dplot")),
            ("GW", params.get("gw")),
            ("FCI", params.get("fci")),
        ]
        for key, value in section_overrides:
            if value:
                input_json.setdefault(key, {})
                _deep_merge(input_json[key], dict(value))

        # ``basis_block`` lets callers override sub-keys of the basis section
        # (e.g. df_basisset) without colliding with the top-level basis string.
        if params.get("basis_block"):
            _deep_merge(input_json["basis"], dict(params["basis_block"]))

        if params.get("exachem_input"):
            # ``restart_from`` is an F10 hint consumed by ``calculate()``; it
            # is not an ExaChem JSON key and must be stripped before merging.
            exachem_input = dict(params["exachem_input"])
            exachem_input.pop("restart_from", None)
            if exachem_input:
                _deep_merge(input_json, exachem_input)

        return input_json

    @staticmethod
    def _locate_output(
        run_dir: Path, input_stem: str, input_json: Dict[str, Any]
    ) -> Tuple[Path, Dict[str, Any]]:
        basis = input_json.get("basis", {}).get("basisset", "")
        scf_type = input_json.get("SCF", {}).get("scf_type", "restricted")

        # ExaChem lays output JSON files under <stem>.<basis>_files/<scf_type>/json/.
        candidate_dir = run_dir / f"{input_stem}.{basis}_files" / scf_type / "json"
        # Prefer the most-derived method file if multiple exist; ExaChem writes
        # one per executed task.
        preferred_order = ("ccsd_t", "ccsd", "mp2", "scf")
        chosen: Optional[Path] = None
        if candidate_dir.is_dir():
            for tag in preferred_order:
                path = candidate_dir / f"{input_stem}.{basis}.{tag}.json"
                if path.is_file():
                    chosen = path
                    break
            if chosen is None:
                json_files = sorted(candidate_dir.glob("*.json"))
                if json_files:
                    chosen = json_files[-1]

        if chosen is None:
            # Fallback: search the whole run dir for any *.scf.json / *.ccsd.json.
            matches = sorted(run_dir.rglob(f"{input_stem}.*.json"))
            matches = [m for m in matches if "/json/" in m.as_posix()]
            if matches:
                chosen = matches[-1]

        if chosen is None:
            raise CalculationFailed(
                f"Could not locate ExaChem JSON output under {run_dir}."
            )

        with open(chosen) as fh:
            payload = json.load(fh)
        return chosen, payload

    @staticmethod
    def _extract_energy(payload: Dict[str, Any], method: str) -> float:
        output = payload.get("output", {})
        if method in ("ccsd_t", "ccsd(t)", "ccsd-t"):
            cc_t = output.get("CCSD(T)", {})
            t_energies = cc_t.get("(T)Energies") or cc_t.get("[T]Energies")
            if t_energies and "total" in t_energies:
                return float(t_energies["total"])
        if method == "ccsd":
            ccsd = output.get("CCSD", {}).get("final_energy", {})
            if isinstance(ccsd, dict) and "total" in ccsd:
                return float(ccsd["total"])
        if method == "mp2":
            mp2 = output.get("MP2", {})
            if "final_energy" in mp2:
                value = mp2["final_energy"]
                return float(value["total"] if isinstance(value, dict) else value)
        scf = output.get("SCF", {})
        if "final_energy" in scf:
            return float(scf["final_energy"])
        raise CalculationFailed(
            f"ExaChem output did not contain a recognized energy for method "
            f"'{method}'. Available keys: {sorted(output)}"
        )

    @staticmethod
    def _extract_components(
        payload: Dict[str, Any], method: str
    ) -> Dict[str, Any]:
        """Pull SCF / MP2 / CCSD / (T) energy components and method metadata.

        Returns a flat dict suitable for merging into the per-row result
        record. Energies are converted to eV; timings to seconds. Missing
        components return ``None`` (e.g. ``mp2_correlation_eV`` for a pure
        SCF run).

        Field naming matches what the JSONL/SQLite consumers expect:
            scf_energy_eV, mp2_correlation_eV, ccsd_correlation_eV,
            t_correction_eV, total_energy_eV,
            scf_time_s, ccsd_time_s, t_time_s,
            basis, scf_type, method, frozen_core.
        """

        output = payload.get("output") or {}
        input_section = payload.get("input") or {}

        scf_block = output.get("SCF") or {}
        ccsd_block = output.get("CCSD") or {}
        ccsd_t_block = output.get("CCSD(T)") or {}
        mp2_block = output.get("MP2") or {}

        def _ha_to_ev(value):
            if value is None:
                return None
            try:
                return float(value) * _HARTREE_TO_EV
            except (TypeError, ValueError):
                return None

        def _seconds(perf):
            if not isinstance(perf, dict):
                return None
            value = perf.get("total_time")
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        # ---- SCF ---------------------------------------------------------
        scf_energy_ha = scf_block.get("final_energy")
        scf_energy_eV = _ha_to_ev(scf_energy_ha)
        scf_time_s = _seconds(scf_block.get("performance"))

        # ---- CCSD --------------------------------------------------------
        ccsd_correlation_eV = None
        ccsd_total_ha = None
        ccsd_final = ccsd_block.get("final_energy")
        if isinstance(ccsd_final, dict):
            ccsd_correlation_eV = _ha_to_ev(ccsd_final.get("correlation"))
            ccsd_total_ha = ccsd_final.get("total")
        ccsd_time_s = _seconds(ccsd_block.get("performance"))

        # ---- (T) ---------------------------------------------------------
        # ExaChem reports both [T] and (T); the canonical "(T)" correction is
        # the asymmetric one — it's what CCSD(T) total energies use.
        t_correction_eV = None
        t_total_ha = None
        t_energies = ccsd_t_block.get("(T)Energies") or {}
        if isinstance(t_energies, dict):
            t_correction_eV = _ha_to_ev(t_energies.get("correction"))
            t_total_ha = t_energies.get("total")
        t_time_s = _seconds(ccsd_t_block.get("performance"))

        # ---- MP2 ---------------------------------------------------------
        # ExaChem's MP2 task output has not been observed in the reference set,
        # but the existing _extract_energy code reads MP2.final_energy. Mirror
        # that path: it may be a scalar (total energy) or a dict containing
        # correlation/total. We surface both correlation (when given) and the
        # total for use as total_energy when method == mp2.
        mp2_correlation_eV = None
        mp2_total_ha = None
        mp2_final = mp2_block.get("final_energy")
        if isinstance(mp2_final, dict):
            mp2_correlation_eV = _ha_to_ev(mp2_final.get("correlation"))
            mp2_total_ha = mp2_final.get("total")
        elif mp2_final is not None:
            mp2_total_ha = mp2_final

        # ---- Total energy (matches highest level requested) --------------
        if method in ("ccsd_t", "ccsd(t)", "ccsd-t") and t_total_ha is not None:
            total_energy_eV = _ha_to_ev(t_total_ha)
        elif method == "ccsd" and ccsd_total_ha is not None:
            total_energy_eV = _ha_to_ev(ccsd_total_ha)
        elif method == "mp2" and mp2_total_ha is not None:
            total_energy_eV = _ha_to_ev(mp2_total_ha)
        else:
            total_energy_eV = scf_energy_eV

        # ---- Method metadata (prefer input echo; fall back to molecule) --
        molecule_section = payload.get("molecule") or {}
        basis = (
            (input_section.get("basis") or {}).get("basisset")
            or (molecule_section.get("basis") or {}).get("basisset")
        )

        scf_input = input_section.get("SCF") or {}
        scf_type = scf_input.get("scf_type")

        # Canonical method label: collapse the various ccsd_t aliases.
        method_label = method
        if method in ("ccsd(t)", "ccsd-t"):
            method_label = "ccsd_t"
        elif method == "hf":
            method_label = "scf"

        # Frozen-core: CC.freeze is a dict like {"atomic": true, "core": N,
        # "virtual": N}. Treat any explicit truthy "atomic" or non-zero
        # "core"/"virtual" as frozen.
        cc_input = input_section.get("CC") or {}
        freeze_block = cc_input.get("freeze")
        if isinstance(freeze_block, dict):
            frozen_core = bool(
                freeze_block.get("atomic")
                or freeze_block.get("core")
                or freeze_block.get("virtual")
            )
        else:
            frozen_core = False

        return {
            "scf_energy_eV": scf_energy_eV,
            "mp2_correlation_eV": mp2_correlation_eV,
            "ccsd_correlation_eV": ccsd_correlation_eV,
            "t_correction_eV": t_correction_eV,
            "total_energy_eV": total_energy_eV,
            "scf_time_s": scf_time_s,
            "ccsd_time_s": ccsd_time_s,
            "t_time_s": t_time_s,
            "basis": basis,
            "scf_type": scf_type,
            "method": method_label,
            "frozen_core": frozen_core,
        }

    # Keys written into ``self.results`` by ``_extract_components`` that the
    # ASE-results-to-row plumbing in ``iqc.asetools`` should copy onto the
    # top-level per-row dict so the orchestrator can persist them.
    EXACHEM_RESULT_FIELDS: Tuple[str, ...] = (
        "scf_energy_eV",
        "mp2_correlation_eV",
        "ccsd_correlation_eV",
        "t_correction_eV",
        "total_energy_eV",
        "scf_time_s",
        "ccsd_time_s",
        "t_time_s",
        "basis",
        "scf_type",
        "method",
        "frozen_core",
    )

    # Artifact retention fields added by F4 / populated by F9. Always present
    # on a successful run so downstream JSONL/SQLite schemas are stable; null
    # / empty when --keep-artifacts is False or F9 is inactive.
    EXACHEM_ARTIFACT_FIELDS: Tuple[str, ...] = (
        "run_dir",
        "artifact_manifest",
        "keep_artifacts",
        "artifact_archive",
    )

    @staticmethod
    def _tail(path: Path, n: int) -> str:
        try:
            with open(path) as fh:
                lines = fh.readlines()
        except OSError:
            return ""
        return "".join(lines[-n:])

    # Patterns (case-insensitive fnmatch) that classify per-file artifacts.
    # Order matters: first match wins so that an MO file named ``*.mo`` does
    # not get tagged ``other`` because it lacks the basename hints.
    _ARTIFACT_PATTERN_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("mo", ("*.mo", "*.mo[0-9]*", "*movecs*", "*.molden", "*orbitals*")),
        ("amplitudes", ("*t1*", "*t2*", "*ccsd_t*amps*", "*amplitudes*")),
        ("cholesky", ("*chol*", "*cd_vec*", "*cholesky*")),
        ("restart", ("*restart*", "*chkpt*", "*checkpoint*")),
        ("output", ("*.json", "*.log", "*.out")),
    )

    @classmethod
    def _classify_artifact(cls, name: str) -> str:
        """Return the manifest 'kind' for a filename based on ExaChem conventions."""

        lowered = name.lower()
        for kind, patterns in cls._ARTIFACT_PATTERN_RULES:
            for pat in patterns:
                if fnmatch.fnmatch(lowered, pat):
                    return kind
        return "other"

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Stream a sha256 so multi-GB amplitude files don't blow the heap."""

        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def _build_artifact_manifest(
        cls,
        run_dir: Path,
        output_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Walk ``run_dir`` and return a list of artifact records.

        Each record is ``{path, kind, size_bytes, sha256}`` with an absolute
        ``path``. The walk is bounded to the supplied ``run_dir`` to avoid
        following stray symlinks into the rest of the filesystem.
        """

        manifest: List[Dict[str, Any]] = []
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            return manifest

        output_resolved = output_path.resolve() if output_path else None

        for entry in sorted(run_dir.rglob("*")):
            if not entry.is_file() or entry.is_symlink():
                continue
            try:
                size_bytes = entry.stat().st_size
            except OSError:
                continue
            resolved = entry.resolve()
            kind = cls._classify_artifact(entry.name)
            # The known ExaChem output JSON is unambiguously 'output' even when
            # the basename also matches an earlier rule.
            if output_resolved is not None and resolved == output_resolved:
                kind = "output"
            try:
                sha = cls._sha256_file(resolved)
            except OSError:
                continue
            manifest.append(
                {
                    "path": str(resolved),
                    "kind": kind,
                    "size_bytes": int(size_bytes),
                    "sha256": sha,
                }
            )
        return manifest
