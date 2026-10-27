"""IQC MCP server.

Exposes a focused subset of IQC's Python API as MCP tools so an LLM agent
can drive quantum-chemistry workflows (build structures from SMILES, run
single-point / opt / vib / thermo / IR, inspect tabular inputs).

Run with:

    iqc-mcp                  # stdio transport (Claude Desktop, Claude Code)
    iqc-mcp --transport sse  # SSE (host/port from FastMCP defaults)

Calculators are loaded lazily inside each tool; importing this module does
not pull in torch/MACE/UMA. The default calculator is `xtb` because it is
small, fast, and supports dipoles (so the IR tool works out of the box).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import threading
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit(
        "iqc-mcp requires the `mcp` package. Install with: "
        "pip install 'iqc[mcp]'  (or: pip install mcp)"
    ) from exc


logger = logging.getLogger("iqc.mcp")

# ASE calculators have mutable state and some backends print to stdout.
# FastMCP captures the protocol stream before invoking these wrappers.
_TOOL_LOCK = threading.RLock()


def _quiet_tool(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _TOOL_LOCK, redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)

    return wrapped


mcp = FastMCP(
    name="iqc",
    instructions=(
        "IQC (Interactive Quantum Chemistry) MCP server. Use these tools to "
        "build molecules from SMILES or XYZ and run single-point, geometry "
        "optimization, vibrational, thermochemistry, or IR calculations with "
        "an ASE calculator (see list_calculators). For IR, the dipole "
        "calculator must support dipoles — use xtb, orca, pyscf HF/DFT, or "
        "mace-polar. Heavy calculators (MACE, UMA) require optional `mlip` "
        "extras; xtb requires the `xtb` extra."
    ),
)


# Calculators that are cheap to (re)instantiate per call. For others (MACE,
# UMA) we cache to avoid re-downloading checkpoints in the same process.
_CALC_CACHE: dict[str, Any] = {}

# Result lists longer than this (e.g. a full ~3700-point IR spectrum) are
# omitted unless the caller asks for arrays, so responses fit agent context.
_MAX_LIST_LEN = 1000

# Paths into a tool's TemporaryDirectory are gone by the time we respond.
_TEMP_PATH_KEYS = {"ir_plot", "normal_mode_data"}


def _jsonable(value, non_finite: list):
    """Convert a result value to strict-JSON data, element by element.

    numpy arrays/scalars become lists/numbers (no scientific imports here, so
    startup stays cheap). Purely real or purely imaginary complex values
    (ASE frequencies and mode energies) follow asetools' frequency
    convention: imaginary modes become negative reals. A complex value with
    both parts nonzero is returned as {"real": x, "imag": y} so neither part
    is lost. NaN/Inf become None and are flagged in `non_finite` so a caller
    can tell missing from zero.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        non_finite.append(True)
        return None
    if isinstance(value, complex):
        if not value.imag:
            return _jsonable(value.real, non_finite)
        if not value.real:
            return _jsonable(-abs(value.imag), non_finite)
        return {
            "real": _jsonable(value.real, non_finite),
            "imag": _jsonable(value.imag, non_finite),
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v, non_finite) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, non_finite) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist(), non_finite)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _trim_results(results: dict[str, Any], keep_arrays: bool = False) -> dict[str, Any]:
    """Drop bulky/non-serializable fields and large arrays from a result dict.

    `forces`, `vib_modes`, and `jmol_vib_modes_xyz` can be huge for big
    molecules. We surface summary fields (energies, frequencies, timings,
    errors, warnings) and keep arrays only when the caller asks. Lists longer
    than `_MAX_LIST_LEN` are omitted unless `keep_arrays`, with their lengths
    reported in `omitted_fields`.
    """

    drop_keys = {"vib_modes", "jmol_vib_modes_xyz"} | _TEMP_PATH_KEYS
    if not keep_arrays:
        drop_keys |= {"forces", "opt_forces"}
    out: dict[str, Any] = {}
    non_finite_fields = []
    omitted_fields = {}
    for k, v in results.items():
        if k in drop_keys:
            continue
        n = len(v) if isinstance(v, (list, tuple)) or getattr(v, "ndim", 0) else 0
        if not keep_arrays and n > _MAX_LIST_LEN:
            omitted_fields[str(k)] = n
            continue
        non_finite: list = []
        out[k] = _jsonable(v, non_finite)
        if non_finite:
            non_finite_fields.append(str(k))
    if non_finite_fields:
        # NaN/Inf were replaced by null; surface which fields were affected.
        out["non_finite_fields"] = non_finite_fields
    if omitted_fields:
        out["omitted_fields"] = omitted_fields
    return out


def _atoms_summary(atoms) -> dict[str, Any]:
    if atoms is None:
        return {}
    return {
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
    }


def _load_atoms(smiles: Optional[str], xyz: Optional[str]):
    """Build an ASE Atoms object from a SMILES string, an XYZ string, or an
    XYZ file path. Exactly one of `smiles`/`xyz` must be set.
    """

    from iqc.asetools import get_atoms_from_smiles, get_atoms_from_xyz

    if smiles and xyz:
        raise ValueError("Provide either 'smiles' or 'xyz', not both.")
    if not smiles and not xyz:
        raise ValueError("One of 'smiles' or 'xyz' is required.")
    if smiles:
        return get_atoms_from_smiles(smiles)
    return get_atoms_from_xyz(xyz)


def _get_calculator(name: str, params: Optional[dict] = None):
    """Cache in-process ML models; file-backed calculators get fresh instances."""

    from iqc.asetools import get_calculator

    params = params or {}
    name = name.strip().lower()  # get_calculator is case-insensitive too
    cache_key = f"{name}:" + json.dumps(params, sort_keys=True, default=str)
    cacheable = name.startswith(("mace", "uma"))
    if cacheable and cache_key in _CALC_CACHE:
        return _CALC_CACHE[cache_key]
    calc = get_calculator(name=name, **params)
    if cacheable:
        _CALC_CACHE[cache_key] = calc
    return calc


@contextmanager
def _orca_scratch(*calculators):
    """Run ORCA calculators in temporary work directories removed on exit.

    asetools otherwise names ORCA directories after `unique_name` in the
    current directory; with a fresh name per MCP call they would accumulate.
    A `directory` passed explicitly in calculator_params is left alone.
    """

    from iqc.asetools import _is_orca_calculator

    with ExitStack() as stack:
        seen = set()
        for calc in calculators:
            if not _is_orca_calculator(calc) or id(calc) in seen:
                continue
            seen.add(id(calc))
            if Path(getattr(calc, "directory", ".")) != Path("."):
                continue
            calc.directory = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="iqc_mcp_orca_")
            )
        yield


# --------------------------------------------------------------------------- #
# Structure tools
# --------------------------------------------------------------------------- #


@mcp.tool()
@_quiet_tool
def smiles_to_xyz(smiles: str) -> dict:
    """Build a 3D structure from a SMILES string and return its XYZ text.

    Uses RDKit's MMFF94 (or UFF fallback) to embed and relax the lowest-energy
    conformer. This is a cheap geometry guess suitable as input to opt / vib.

    Args:
        smiles: SMILES string, e.g. "CCO" for ethanol.

    Returns:
        {"xyz": "...", "n_atoms": N, "formula": "C2H6O", "canonical_smiles": "..."}
    """

    from iqc.asetools import atoms2xyz, get_atoms_from_smiles

    atoms = get_atoms_from_smiles(smiles)
    return {
        "xyz": atoms2xyz(atoms),
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "canonical_smiles": atoms.info.get("canonical_smiles", ""),
    }


@mcp.tool()
@_quiet_tool
def xyz_to_smiles(xyz: str) -> dict:
    """Derive a canonical SMILES from XYZ coordinates (RDKit perception).

    Args:
        xyz: XYZ-format string or path to an .xyz file.

    Returns:
        {"smiles": "...", "n_atoms": N, "formula": "..."}
    """

    from iqc.asetools import get_atoms_from_xyz, get_canonical_smiles_from_atoms

    atoms = get_atoms_from_xyz(xyz)
    return {
        "smiles": get_canonical_smiles_from_atoms(atoms),
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
    }


@mcp.tool()
@_quiet_tool
def list_calculators() -> dict:
    """List ASE calculator names IQC's MCP tools accept, with notes."""

    return {
        "calculators": [
            {
                "name": "xtb",
                "type": "semi-empirical",
                "dipole": True,
                "notes": "Fast GFN2-xTB. Requires `pip install iqc[xtb]`.",
            },
            {
                "name": "emt",
                "type": "classical",
                "dipole": False,
                "notes": "Effective-medium theory; metals only. No dipoles.",
            },
            {
                "name": "mace",
                "type": "MLIP foundation model",
                "dipole": False,
                "notes": "MACE-MP large. Requires `iqc[mlip]` + mace-torch.",
            },
            {
                "name": "mace-polar",
                "type": "MLIP (electrostatic)",
                "dipole": True,
                "notes": "Electrostatic MACE; supports IR. Needs MACE main branch.",
            },
            {
                "name": "uma-s-omol",
                "type": "MLIP (FAIRChem)",
                "dipole": False,
                "notes": "UMA small, organic-molecule head.",
            },
            {
                "name": "uma-m-omol",
                "type": "MLIP (FAIRChem)",
                "dipole": False,
                "notes": "UMA medium, organic-molecule head.",
            },
            {
                "name": "orca",
                "type": "DFT/HF",
                "dipole": True,
                "notes": "ORCA binary must be on PATH or pass calculator_params.command.",
            },
            {
                "name": "pyscf",
                "type": "DFT/HF",
                "dipole": True,
                "notes": (
                    "Requires pyscf. Set method, xc, and basis explicitly. "
                    "Dipoles (IR) only for HF/DFT; MP2/CCSD methods are "
                    "energy-only."
                ),
            },
            {
                "name": "pyscf-ccsd(t)",
                "type": "correlated",
                "dipole": False,
                "notes": "Requires pyscf; expensive finite-difference forces.",
            },
            {
                "name": "exachem",
                "type": "external quantum chemistry",
                "dipole": False,
                "notes": "Requires ExaChem executable; energy-only methods use FD forces.",
            },
            {
                "name": "vasp",
                "type": "plane-wave DFT",
                "dipole": False,
                "notes": "Requires licensed VASP and pseudopotentials; molecular box workflow.",
            },
        ],
        "default_for_mcp": "xtb",
        "ir_compatible": ["xtb", "orca", "mace-polar", "pyscf"],
    }


# --------------------------------------------------------------------------- #
# Computation tools
# --------------------------------------------------------------------------- #


@mcp.tool()
@_quiet_tool
def run_single_point(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    keep_forces: bool = False,
) -> dict:
    """Compute single-point energy (and forces if available).

    Args:
        smiles: SMILES string of the molecule. Mutually exclusive with `xyz`.
        xyz: XYZ-format string or path to an .xyz file.
        calculator: Calculator name (see `list_calculators`).
        calculator_params: Extra kwargs forwarded to the calculator constructor
            (e.g. {"orcasimpleinput": "B3LYP def2-SVP"} for ORCA).
        charge: Net molecular charge; omitted uses input metadata, then zero.
        multiplicity: Spin multiplicity 2S+1; defaults from electron count.
        keep_forces: Include the full force array in the response.

    Returns:
        Dict with `energy_eV`, `formula`, `calc_time`, error/warnings, and
        `forces` when `keep_forces=True`.
    """

    from iqc.asetools import run_single_point as _rsp

    atoms = _load_atoms(smiles, xyz)
    calc = _get_calculator(calculator, calculator_params)
    with _orca_scratch(calc):
        atoms, results = _rsp(
            atoms,
            calculator=calc,
            unique_name="mcp_single_" + os.urandom(6).hex(),
            charge=charge,
            multiplicity=multiplicity,
        )
    out = _trim_results(results, keep_arrays=keep_forces)
    out["formula"] = atoms.get_chemical_formula() if atoms is not None else ""
    return out


@mcp.tool()
@_quiet_tool
def run_optimization(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    fmax: float = 0.01,
    max_steps: int = 200,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
) -> dict:
    """Optimize geometry with BFGS to the requested force tolerance.

    Args:
        smiles | xyz: Input structure (exactly one).
        calculator: See `list_calculators`.
        calculator_params: Extra kwargs for the calculator.
        fmax: Force convergence threshold in eV/Å.
        max_steps: BFGS step cap.
        charge: Net molecular charge; omitted uses input metadata, then zero.
        multiplicity: Spin multiplicity 2S+1.

    Returns:
        Dict including `opt_energy_eV`, `opt_xyz`, `opt_smiles`,
        `opt_converged`, `opt_steps`, `smiles_changed`, timings, and errors.
    """

    from iqc.asetools import run_optimization as _ropt

    atoms = _load_atoms(smiles, xyz)
    calc = _get_calculator(calculator, calculator_params)
    with _orca_scratch(calc):
        atoms, results = _ropt(
            atoms,
            calculator=calc,
            unique_name="mcp_opt_" + os.urandom(6).hex(),
            fmax=fmax,
            max_steps=max_steps,
            charge=charge,
            multiplicity=multiplicity,
        )
    return _trim_results(results)


@mcp.tool()
@_quiet_tool
def run_vibrations(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    optimize: bool = True,
    delta: float = 0.01,
    fmax: float = 0.01,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    keep_arrays: bool = False,
) -> dict:
    """Finite-difference vibrational analysis.

    Args:
        smiles | xyz: Input structure.
        calculator: See `list_calculators`.
        calculator_params: Extra calculator kwargs.
        optimize: Pre-optimize geometry (recommended; imaginary modes appear
            if you skip this and the input is not a stationary point).
        delta: Finite-difference displacement in Å.
        fmax: Pre-optimization force tolerance.
        charge, multiplicity: Charge and spin state.
        keep_arrays: Include arrays longer than 1000 entries; otherwise
            their lengths are reported in `omitted_fields`.

    Returns:
        Dict with `vibrational_frequencies_cm^-1`, `number_of_imaginary`,
        `vib_time`, plus the embedded optimization results. Vibrational
        modes/Jmol output are omitted to keep the response small.
    """

    from iqc.asetools import run_vibrations as _rvib

    atoms = _load_atoms(smiles, xyz)
    calc = _get_calculator(calculator, calculator_params)
    with _orca_scratch(calc), tempfile.TemporaryDirectory(
        prefix="iqc_mcp_vib_"
    ) as vib_dir:
        atoms, results = _rvib(
            atoms,
            calculator=calc,
            optimize=optimize,
            unique_name="mcp_vib_" + os.urandom(6).hex(),
            vib_dir=vib_dir,
            delta=delta,
            fmax=fmax,
            charge=charge,
            multiplicity=multiplicity,
        )
    return _trim_results(results, keep_arrays=keep_arrays)


@mcp.tool()
@_quiet_tool
def run_thermo(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    ignore_imag_modes: bool = True,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    keep_arrays: bool = False,
) -> dict:
    """Ideal-gas thermochemistry at 298.15 K / 1 atm.

    Runs opt + vibrations, then evaluates Gibbs free energy, enthalpy,
    entropy, and ZPE correction with ASE's IdealGasThermo.

    Args:
        keep_arrays: Include arrays longer than 1000 entries; otherwise
            their lengths are reported in `omitted_fields`.

    Returns:
        Dict including `G_eV`, `H_eV`, `S_eV/K`, `E_ZPE_eV`, plus the
        vibrational and optimization fields.
    """

    from iqc.asetools import run_thermo as _rth

    atoms = _load_atoms(smiles, xyz)
    calc = _get_calculator(calculator, calculator_params)
    with _orca_scratch(calc), tempfile.TemporaryDirectory(
        prefix="iqc_mcp_thermo_"
    ) as vib_dir:
        thermo, results = _rth(
            atoms,
            calculator=calc,
            ignore_imag_modes=ignore_imag_modes,
            unique_name="mcp_thermo_" + os.urandom(6).hex(),
            vib_dir=vib_dir,
            charge=charge,
            multiplicity=multiplicity,
        )
    return _trim_results(results, keep_arrays=keep_arrays)


@mcp.tool()
@_quiet_tool
def run_ir(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    optimization_calculator: Optional[str] = None,
    vibration_calculator: Optional[str] = None,
    dipole_calculator: Optional[str] = None,
    optimization_calculator_params: Optional[dict] = None,
    vibration_calculator_params: Optional[dict] = None,
    dipole_calculator_params: Optional[dict] = None,
    optimize: bool = True,
    delta: float = 0.01,
    fmax: float = 0.01,
    sparse_spectrum: bool = True,
    intensity_threshold: float = 1e-3,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    keep_arrays: bool = False,
) -> dict:
    """Infrared spectrum (frequencies + intensities) by finite differences.

    Three roles (optimization, vibration, dipole) can use different
    calculators. Roles without an override use `calculator`, which is only
    constructed when some role needs it. The **dipole** role must be
    a calculator that implements dipoles (`xtb`, `orca`, `mace-polar`, or
    `pyscf` with HF/DFT) — `mace`, `emt`, `uma-*`, and correlated PySCF
    methods fail before any finite-difference work starts.

    Args:
        smiles | xyz: Input structure.
        calculator: Fallback calculator for all three roles.
        calculator_params: Extra kwargs for `calculator`. A role override
            naming the same calculator inherits them; a role naming a
            different calculator gets none unless its `*_calculator_params`
            is set.
        optimization_calculator, vibration_calculator, dipole_calculator:
            Per-role overrides (calculator name).
        optimization_calculator_params, vibration_calculator_params,
        dipole_calculator_params: Per-role calculator kwargs, e.g.
            dipole_calculator="orca" with
            dipole_calculator_params={"orcasimpleinput": "B3LYP def2-SVP"}
            alongside calculator="mace".
        optimize: Pre-optimize geometry.
        delta: Finite-difference displacement in Å.
        sparse_spectrum: If True, only points above `intensity_threshold`
            are kept in the spectrum arrays.
        intensity_threshold: D/Å²·amu⁻¹ cutoff for `sparse_spectrum`.
        keep_arrays: Return arrays longer than 1000 points (e.g. the full
            spectrum with `sparse_spectrum=False`); otherwise they are listed
            in `omitted_fields` with their lengths.

    Returns:
        Dict with `spectrum_frequencies`, `spectrum_intensities`,
        `vibrational_frequencies_cm^-1`, and the embedded opt fields.
    """

    from iqc.asetools import run_ir as _rir

    atoms = _load_atoms(smiles, xyz)

    def same_name(a, b):
        return bool(a and b) and a.strip().lower() == b.strip().lower()

    def role(name, params):
        if not name:
            return None
        if params is None:
            # calculator_params belong to `calculator`; don't hand ORCA
            # kwargs to xtb or MACE just because a role was overridden.
            params = calculator_params if same_name(name, calculator) else None
        return _get_calculator(name, params)

    opt_c = (
        role(optimization_calculator, optimization_calculator_params)
        if optimize
        else None
    )
    vib_c = role(vibration_calculator, vibration_calculator_params)
    dip_c = role(dipole_calculator, dipole_calculator_params)
    # Build the fallback only if a role uses it, so a fully overridden run
    # does not require the default backend (xtb) to be installed.
    needs_fallback = (optimize and opt_c is None) or vib_c is None or dip_c is None
    fallback = (
        _get_calculator(calculator, calculator_params)
        if calculator and needs_fallback
        else None
    )

    with _orca_scratch(fallback, opt_c, vib_c, dip_c), tempfile.TemporaryDirectory(
        prefix="iqc_mcp_ir_"
    ) as vib_dir:
        atoms, results = _rir(
            atoms,
            calculator=fallback,
            optimization_calculator=opt_c,
            vibration_calculator=vib_c,
            dipole_calculator=dip_c,
            optimize=optimize,
            unique_name="mcp_ir_" + os.urandom(6).hex(),
            vib_dir=vib_dir,
            delta=delta,
            fmax=fmax,
            sparse_spectrum=sparse_spectrum,
            intensity_threshold=intensity_threshold,
            charge=charge,
            multiplicity=multiplicity,
        )
    return _trim_results(results, keep_arrays=keep_arrays)


# --------------------------------------------------------------------------- #
# Data tools
# --------------------------------------------------------------------------- #


@mcp.tool()
@_quiet_tool
def inspect_data_file(path: str) -> dict:
    """Inspect a tabular file (parquet, CSV/TSV, Excel, JSON/JSONL, Feather,
    Arrow IPC) and return its schema and basic statistics.

    Useful before running batch IQC calculations from a results file to pick
    the right SMILES or XYZ column.
    """

    from iqc.datatools import inspect_data_file as _inspect

    summary = _inspect(path)
    payload = asdict(summary) if is_dataclass(summary) else summary
    payload["path"] = str(payload["path"])
    return payload


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IQC MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("IQC_MCP_LOG", "WARNING"),
        help="Python logging level (default: WARNING).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    # Avoid initializing MPI for the long-lived server process.
    os.environ.setdefault("IQC_DISABLE_MPI", "1")

    if args.transport == "stdio":
        # Keep a dedicated protocol descriptor. Native libraries and child
        # processes writing to fd 1 must not corrupt JSON-RPC on stdout.
        protocol_stdout = os.fdopen(
            os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1
        )
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = protocol_stdout
        # Likewise keep JSON-RPC input off fd 0, so child processes (mpirun,
        # file-IO engines) that read stdin cannot consume client requests.
        protocol_stdin = os.fdopen(os.dup(sys.stdin.fileno()), "r", encoding="utf-8")
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, sys.stdin.fileno())
        os.close(devnull)
        sys.stdin = protocol_stdin

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
