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
import os
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit(
        "iqc-mcp requires the `mcp` package. Install with: "
        "pip install 'iqc[mcp]'  (or: pip install mcp)"
    ) from exc


logger = logging.getLogger("iqc.mcp")

mcp = FastMCP(
    name="iqc",
    instructions=(
        "IQC (Interactive Quantum Chemistry) MCP server. Use these tools to "
        "build molecules from SMILES or XYZ and run single-point, geometry "
        "optimization, vibrational, thermochemistry, or IR calculations with "
        "an ASE calculator (xtb, mace, mace-polar, emt, orca, uma-*). For IR, "
        "the dipole calculator must support dipoles — use xtb, orca, or "
        "mace-polar. Heavy calculators (MACE, UMA) require optional `mlip` "
        "extras; xtb requires the `xtb` extra."
    ),
)


# Calculators that are cheap to (re)instantiate per call. For others (MACE,
# UMA) we cache to avoid re-downloading checkpoints in the same process.
_CALC_CACHE: dict[str, Any] = {}


def _trim_results(results: dict[str, Any], keep_arrays: bool = False) -> dict[str, Any]:
    """Drop bulky/non-serializable fields and large arrays from a result dict.

    `forces`, `vib_modes`, and `jmol_vib_modes_xyz` can be huge for big
    molecules. We surface summary fields (energies, frequencies, timings,
    errors, warnings) and keep arrays only when the caller asks.
    """

    drop_keys = {"vib_modes", "jmol_vib_modes_xyz"}
    if not keep_arrays:
        drop_keys |= {"forces", "opt_forces"}
    out: dict[str, Any] = {}
    for k, v in results.items():
        if k in drop_keys:
            continue
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            v = repr(v)
        out[k] = v
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
    """Cache calculator instances by (name, frozen params)."""

    from iqc.asetools import get_calculator

    params = params or {}
    cache_key = f"{name}:" + json.dumps(params, sort_keys=True, default=str)
    if cache_key in _CALC_CACHE:
        return _CALC_CACHE[cache_key]
    calc = get_calculator(name=name, **params)
    _CALC_CACHE[cache_key] = calc
    return calc


# --------------------------------------------------------------------------- #
# Structure tools
# --------------------------------------------------------------------------- #


@mcp.tool()
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
def list_calculators() -> dict:
    """List ASE calculator names IQC's MCP tools accept, with notes."""

    return {
        "calculators": [
            {"name": "xtb", "type": "semi-empirical", "dipole": True,
             "notes": "Fast GFN2-xTB. Requires `pip install iqc[xtb]`."},
            {"name": "emt", "type": "classical", "dipole": False,
             "notes": "Effective-medium theory; metals only. No dipoles."},
            {"name": "mace", "type": "MLIP foundation model", "dipole": False,
             "notes": "MACE-MP large. Requires `iqc[mlip]` + mace-torch."},
            {"name": "mace-polar", "type": "MLIP (electrostatic)", "dipole": True,
             "notes": "Electrostatic MACE; supports IR. Needs MACE main branch."},
            {"name": "uma-s-omol", "type": "MLIP (FAIRChem)", "dipole": False,
             "notes": "UMA small, organic-molecule head."},
            {"name": "uma-m-omol", "type": "MLIP (FAIRChem)", "dipole": False,
             "notes": "UMA medium, organic-molecule head."},
            {"name": "orca", "type": "DFT/HF", "dipole": True,
             "notes": "ORCA binary must be on PATH or pass calculator_params.command."},
        ],
        "default_for_mcp": "xtb",
        "ir_compatible": ["xtb", "orca", "mace-polar"],
    }


# --------------------------------------------------------------------------- #
# Computation tools
# --------------------------------------------------------------------------- #


@mcp.tool()
def run_single_point(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    charge: int = 0,
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
        charge: Net molecular charge.
        multiplicity: Spin multiplicity 2S+1; defaults from electron count.
        keep_forces: Include the full force array in the response.

    Returns:
        Dict with `energy_eV`, `formula`, `calc_time`, error/warnings, and
        `forces` when `keep_forces=True`.
    """

    from iqc.asetools import run_single_point as _rsp

    atoms = _load_atoms(smiles, xyz)
    calc = _get_calculator(calculator, calculator_params)
    atoms, results = _rsp(
        atoms,
        calculator=calc,
        unique_name="mcp_single",
        charge=charge,
        multiplicity=multiplicity,
    )
    out = _trim_results(results, keep_arrays=keep_forces)
    out["formula"] = atoms.get_chemical_formula() if atoms is not None else ""
    return out


@mcp.tool()
def run_optimization(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    fmax: float = 0.01,
    max_steps: int = 200,
    charge: int = 0,
    multiplicity: Optional[int] = None,
) -> dict:
    """Optimize geometry with BFGS to the requested force tolerance.

    Args:
        smiles | xyz: Input structure (exactly one).
        calculator: See `list_calculators`.
        calculator_params: Extra kwargs for the calculator.
        fmax: Force convergence threshold in eV/Å.
        max_steps: BFGS step cap.
        charge: Net molecular charge.
        multiplicity: Spin multiplicity 2S+1.

    Returns:
        Dict including `opt_energy_eV`, `opt_xyz`, `opt_smiles`,
        `opt_converged`, `opt_steps`, `smiles_changed`, timings, and errors.
    """

    from iqc.asetools import run_optimization as _ropt

    atoms = _load_atoms(smiles, xyz)
    calc = _get_calculator(calculator, calculator_params)
    atoms, results = _ropt(
        atoms,
        calculator=calc,
        unique_name="mcp_opt",
        fmax=fmax,
        max_steps=max_steps,
        charge=charge,
        multiplicity=multiplicity,
    )
    return _trim_results(results)


@mcp.tool()
def run_vibrations(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    optimize: bool = True,
    delta: float = 0.01,
    fmax: float = 0.01,
    charge: int = 0,
    multiplicity: Optional[int] = None,
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

    Returns:
        Dict with `vibrational_frequencies_cm^-1`, `number_of_imaginary`,
        `vib_time`, plus the embedded optimization results. Vibrational
        modes/Jmol output are omitted to keep the response small.
    """

    from iqc.asetools import run_vibrations as _rvib

    atoms = _load_atoms(smiles, xyz)
    calc = _get_calculator(calculator, calculator_params)
    with tempfile.TemporaryDirectory(prefix="iqc_mcp_vib_") as vib_dir:
        atoms, results = _rvib(
            atoms,
            calculator=calc,
            optimize=optimize,
            unique_name="mcp_vib",
            vib_dir=vib_dir,
            delta=delta,
            fmax=fmax,
            charge=charge,
            multiplicity=multiplicity,
        )
    return _trim_results(results)


@mcp.tool()
def run_thermo(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    ignore_imag_modes: bool = True,
    charge: int = 0,
    multiplicity: Optional[int] = None,
) -> dict:
    """Ideal-gas thermochemistry at 298.15 K / 1 atm.

    Runs opt + vibrations, then evaluates Gibbs free energy, enthalpy,
    entropy, and ZPE correction with ASE's IdealGasThermo.

    Returns:
        Dict including `G_eV`, `H_eV`, `S_eV/K`, `E_ZPE_eV`, plus the
        vibrational and optimization fields.
    """

    from iqc.asetools import run_thermo as _rth

    atoms = _load_atoms(smiles, xyz)
    calc = _get_calculator(calculator, calculator_params)
    with tempfile.TemporaryDirectory(prefix="iqc_mcp_thermo_") as vib_dir:
        thermo, results = _rth(
            atoms,
            calculator=calc,
            ignore_imag_modes=ignore_imag_modes,
            unique_name="mcp_thermo",
            vib_dir=vib_dir,
            charge=charge,
            multiplicity=multiplicity,
        )
    return _trim_results(results)


@mcp.tool()
def run_ir(
    smiles: Optional[str] = None,
    xyz: Optional[str] = None,
    calculator: str = "xtb",
    calculator_params: Optional[dict] = None,
    optimization_calculator: Optional[str] = None,
    vibration_calculator: Optional[str] = None,
    dipole_calculator: Optional[str] = None,
    optimize: bool = True,
    delta: float = 0.01,
    fmax: float = 0.01,
    sparse_spectrum: bool = True,
    intensity_threshold: float = 1e-3,
    charge: int = 0,
    multiplicity: Optional[int] = None,
) -> dict:
    """Infrared spectrum (frequencies + intensities) by finite differences.

    Three roles (optimization, vibration, dipole) can use different
    calculators. Roles fall back to `calculator`. The **dipole** role must be
    a calculator that implements dipoles (`xtb`, `orca`, or `mace-polar`) —
    `mace`, `emt`, and `uma-*` will fail at the dipole step.

    Args:
        smiles | xyz: Input structure.
        calculator: Fallback calculator for all three roles.
        calculator_params: Extra kwargs applied to whichever calculator
            instances `calculator`-name resolves to (shared by all roles).
        optimization_calculator, vibration_calculator, dipole_calculator:
            Per-role overrides (calculator name).
        optimize: Pre-optimize geometry.
        delta: Finite-difference displacement in Å.
        sparse_spectrum: If True, only points above `intensity_threshold`
            are kept in the spectrum arrays.
        intensity_threshold: D/Å²·amu⁻¹ cutoff for `sparse_spectrum`.

    Returns:
        Dict with `spectrum_frequencies`, `spectrum_intensities`,
        `vibrational_frequencies_cm^-1`, and the embedded opt fields.
    """

    from iqc.asetools import run_ir as _rir

    atoms = _load_atoms(smiles, xyz)
    fallback = _get_calculator(calculator, calculator_params) if calculator else None
    opt_c = _get_calculator(optimization_calculator, calculator_params) if optimization_calculator else None
    vib_c = _get_calculator(vibration_calculator, calculator_params) if vibration_calculator else None
    dip_c = _get_calculator(dipole_calculator, calculator_params) if dipole_calculator else None

    with tempfile.TemporaryDirectory(prefix="iqc_mcp_ir_") as vib_dir:
        atoms, results = _rir(
            atoms,
            calculator=fallback,
            optimization_calculator=opt_c,
            vibration_calculator=vib_c,
            dipole_calculator=dip_c,
            optimize=optimize,
            unique_name="mcp_ir",
            vib_dir=vib_dir,
            delta=delta,
            fmax=fmax,
            sparse_spectrum=sparse_spectrum,
            intensity_threshold=intensity_threshold,
            charge=charge,
            multiplicity=multiplicity,
        )
    return _trim_results(results)


# --------------------------------------------------------------------------- #
# Data tools
# --------------------------------------------------------------------------- #


@mcp.tool()
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
        choices=("stdio", "sse"),
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
    )

    # Avoid initializing MPI for the long-lived server process.
    os.environ.setdefault("IQC_DISABLE_MPI", "1")

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
