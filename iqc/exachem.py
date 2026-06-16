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
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
_DEFAULT_BINARY = "/home/keceli/soft/nwx/exachem/build/install/bin/ExaChem"

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
        run_dir = self._prepare_run_dir(params.get("keep_files", False))
        input_path = run_dir / "input.json"
        stdout_path = run_dir / "exachem.log"

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

        with open(stdout_path, "w") as log_fh:
            completed = subprocess.run(
                cmd,
                cwd=str(run_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )

        self.last_input_path = input_path
        self.last_stdout_path = stdout_path
        self.last_run_dir = run_dir

        if completed.returncode != 0:
            tail = self._tail(stdout_path, 40)
            raise CalculationFailed(
                f"ExaChem exited with code {completed.returncode}. "
                f"Command: {' '.join(cmd)}\nLog tail:\n{tail}"
            )

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_run_dir(self, keep_files: bool) -> Path:
        base = Path(self.directory).resolve()
        base.mkdir(parents=True, exist_ok=True)
        run_dir = base / "exachem_run"
        if run_dir.exists() and not keep_files:
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

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
            _deep_merge(input_json, dict(params["exachem_input"]))

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
    def _tail(path: Path, n: int) -> str:
        try:
            with open(path) as fh:
                lines = fh.readlines()
        except OSError:
            return ""
        return "".join(lines[-n:])
