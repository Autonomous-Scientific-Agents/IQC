"""NMR workflow utilities for IQC."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.io import read, write
from ase.optimize import BFGS
from ase.units import Hartree

from iqc.asetools import atoms2xyz

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds, rdMolDescriptors
except (
    ImportError
):  # pragma: no cover - rdkit is a runtime dependency, but keep graceful fallback
    Chem = None
    AllChem = None
    rdDetermineBonds = None
    rdMolDescriptors = None


DEFAULT_NMR_BACKEND = "orca"
DEFAULT_NMR_METHOD = "PBE0"
DEFAULT_NMR_BASIS = "def2-TZVP"
DEFAULT_OPT_METHOD = "B3LYP"
DEFAULT_OPT_BASIS = "def2-SVP"
DEFAULT_XTB_OPT_METHOD = "GFN2-xTB"
DEFAULT_SOLVENT_MODEL = "smd"
DEFAULT_SOLVENT = "chloroform"
DEFAULT_NUCLEI = ["1H", "13C"]
DEFAULT_TEMPERATURE_K = 298.15
DEFAULT_NUM_CONFORMERS = 10
DEFAULT_RMSD_PRUNE = 0.5
DEFAULT_RELATIVE_ENERGY_WINDOW_KCAL = 5.0
DEFAULT_OPT_FMAX = 0.05
DEFAULT_OPT_STEPS = 250
DEFAULT_GRID_POINTS = 4096
DEFAULT_LINEWIDTHS = {
    "1H": 0.01,
    "13C": 1.0,
}
DEFAULT_PLOT_RANGES = {
    "1H": (-1.0, 12.0),
    "13C": (-10.0, 220.0),
}
# Approximate internal references for routine organic NMR prediction.
# Users should supply calibrated values for production work.
DEFAULT_REFERENCE_SHIELDINGS = {
    "1H": 31.77,
    "13C": 188.10,
}
K_BOLTZMANN_EV_PER_K = 8.617333262145e-5
KCAL_MOL_TO_EV = 0.0433641153087705

NUCLEUS_RE = re.compile(r"^\s*(\d*)([A-Za-z]{1,2})\s*$")

METHOD_ALIASES = {
    "orca": {
        "pbe0": "PBE0",
        "b3lyp": "B3LYP",
        "m06-2x": "M06-2X",
    },
    "gaussian": {
        "pbe0": "PBE1PBE",
        "b3lyp": "B3LYP",
        "m06-2x": "M062X",
    },
    "nwchem": {
        "pbe0": "pbe0",
        "b3lyp": "b3lyp",
        "m06-2x": "m06-2x",
    },
}

BASIS_ALIASES = {
    "orca": {
        "def2-svp": "def2-SVP",
        "def2-tzvp": "def2-TZVP",
    },
    "gaussian": {
        "def2-svp": "def2SVP",
        "def2-tzvp": "def2TZVP",
    },
    "nwchem": {
        "def2-svp": "def2-svp",
        "def2-tzvp": "def2-tzvp",
    },
}

SOLVENT_ALIASES = {
    "chloroform": {
        "orca": "chloroform",
        "gaussian": "chloroform",
        "nwchem": "chloroform",
    },
    "chcl3": {
        "orca": "chloroform",
        "gaussian": "chloroform",
        "nwchem": "chloroform",
    },
    "dmso": {
        "orca": "dmso",
        "gaussian": "dmso",
        "nwchem": "dmso",
    },
    "water": {
        "orca": "water",
        "gaussian": "water",
        "nwchem": "water",
    },
    "methanol": {
        "orca": "methanol",
        "gaussian": "methanol",
        "nwchem": "methanol",
    },
    "acetonitrile": {
        "orca": "acetonitrile",
        "gaussian": "acetonitrile",
        "nwchem": "acetonitrile",
    },
}

XTB_SOLVENT_ALIASES = {
    "chloroform": "chcl3",
    "chcl3": "chcl3",
    "water": "water",
    "methanol": "methanol",
    "acetonitrile": "acetonitrile",
    "acetone": "acetone",
    "benzene": "benzene",
    "dmso": "dmso",
    "thf": "thf",
    "toluene": "toluene",
    "dichloromethane": "ch2cl2",
    "ch2cl2": "ch2cl2",
}


@dataclass
class ConformerCandidate:
    """Geometry candidate used in conformer/NMR workflows."""

    conformer_id: int
    atoms: Atoms
    initial_energy_eV: Optional[float] = None
    energy_source: str = ""
    boltzmann_weight: float = 0.0
    output_dir: Optional[Path] = None
    optimized_xyz: str = ""
    optimization_energy_eV: Optional[float] = None
    optimization_converged: Optional[bool] = None


@dataclass
class NMRSettings:
    """Configuration for the NMR workflow."""

    backend: str = DEFAULT_NMR_BACKEND
    nuclei: List[str] = field(default_factory=lambda: list(DEFAULT_NUCLEI))
    method: str = DEFAULT_NMR_METHOD
    basis: str = DEFAULT_NMR_BASIS
    solvent_model: Optional[str] = DEFAULT_SOLVENT_MODEL
    solvent: Optional[str] = DEFAULT_SOLVENT
    charge: int = 0
    multiplicity: int = 1
    optimize_geometry: bool = True
    conformer_sampling: Optional[bool] = None
    num_conformers: int = DEFAULT_NUM_CONFORMERS
    temperature: float = DEFAULT_TEMPERATURE_K
    linewidth: Optional[float] = None
    plot_range: Optional[Tuple[float, float]] = None
    lineshape: str = "lorentzian"
    output_dir: Optional[str] = None
    reference_shieldings: Dict[str, float] = field(default_factory=dict)
    optimization_backend: Optional[str] = None
    optimization_method: str = DEFAULT_OPT_METHOD
    optimization_basis: str = DEFAULT_OPT_BASIS
    optimization_fmax: float = DEFAULT_OPT_FMAX
    optimization_max_steps: int = DEFAULT_OPT_STEPS
    relative_energy_window_kcal: float = DEFAULT_RELATIVE_ENERGY_WINDOW_KCAL
    rmsd_prune_threshold: float = DEFAULT_RMSD_PRUNE
    voigt_eta: float = 0.5
    command: Optional[str] = None
    calculator_kwargs: Dict[str, object] = field(default_factory=dict)


def _normalize_element(symbol: str) -> str:
    return symbol.strip().capitalize()


def _parse_nucleus_label(label: str) -> Tuple[str, str]:
    match = NUCLEUS_RE.match(str(label))
    if not match:
        raise ValueError(f"Invalid nucleus label '{label}'.")
    isotope, element = match.groups()
    element = _normalize_element(element)
    return isotope + element if isotope else element, element


def normalize_nuclei(nuclei: Optional[Sequence[str]]) -> List[str]:
    """Normalize a user-provided nuclei list."""

    if nuclei is None:
        nuclei = DEFAULT_NUCLEI
    elif isinstance(nuclei, str):
        nuclei = [nuclei]

    normalized: List[str] = []
    for item in nuclei:
        if item is None:
            continue
        parts = [part.strip() for part in str(item).split(",") if part.strip()]
        for part in parts:
            nucleus, _ = _parse_nucleus_label(part)
            normalized.append(nucleus)

    if not normalized:
        normalized = list(DEFAULT_NUCLEI)

    deduped: List[str] = []
    seen = set()
    for nucleus in normalized:
        if nucleus not in seen:
            deduped.append(nucleus)
            seen.add(nucleus)
    return deduped


def parse_reference_shieldings(
    values: Optional[Iterable[object]],
) -> Dict[str, float]:
    """Parse `nucleus=value` reference shielding overrides."""

    if values is None:
        return {}
    if isinstance(values, dict):
        parsed: Dict[str, float] = {}
        for key, value in values.items():
            nucleus, _ = _parse_nucleus_label(str(key))
            parsed[nucleus] = float(value)
        return parsed

    parsed = {}
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Reference shielding '{item}' must use the form nucleus=value."
            )
        nucleus_text, shielding_text = item.split("=", 1)
        nucleus, _ = _parse_nucleus_label(nucleus_text)
        parsed[nucleus] = float(shielding_text)
    return parsed


def build_nmr_settings(**params) -> NMRSettings:
    """Construct settings with sane defaults and normalized inputs."""

    settings = NMRSettings(**params)
    settings.backend = settings.backend.lower()
    if settings.backend == "xtb":
        raise ValueError(
            "xTB is not supported as a direct NMR backend. "
            "Use backend=orca/nwchem/gaussian and optimization_backend=xtb instead."
        )
    if settings.backend not in {"orca", "nwchem", "gaussian"}:
        raise ValueError(
            f"Unsupported NMR backend '{settings.backend}'. Use orca, nwchem, or gaussian."
        )
    if settings.optimization_backend is not None:
        settings.optimization_backend = settings.optimization_backend.lower()
        if settings.optimization_backend not in {"orca", "nwchem", "gaussian", "xtb"}:
            raise ValueError(
                "Unsupported optimization backend. Use orca, nwchem, gaussian, or xtb."
            )
    if _optimization_backend(settings) == "xtb":
        method_key = (settings.optimization_method or "").strip().lower()
        if method_key in {"", DEFAULT_OPT_METHOD.lower()}:
            settings.optimization_method = DEFAULT_XTB_OPT_METHOD
        _xtb_method_args(settings.optimization_method)

    settings.nuclei = normalize_nuclei(settings.nuclei)
    settings.reference_shieldings = parse_reference_shieldings(
        settings.reference_shieldings
    )
    settings.lineshape = settings.lineshape.lower()
    if settings.lineshape not in {"lorentzian", "gaussian", "pseudo-voigt"}:
        raise ValueError(
            "Unsupported lineshape. Use lorentzian, gaussian, or pseudo-voigt."
        )
    if settings.num_conformers < 1:
        raise ValueError("num_conformers must be at least 1.")
    if settings.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if settings.multiplicity < 1:
        raise ValueError("multiplicity must be at least 1.")
    if settings.plot_range is not None:
        settings.plot_range = (
            float(settings.plot_range[0]),
            float(settings.plot_range[1]),
        )
    return settings


def _resolve_method(method: str, backend: str) -> str:
    return METHOD_ALIASES.get(backend, {}).get(method.lower(), method)


def _resolve_basis(basis: str, backend: str) -> str:
    return BASIS_ALIASES.get(backend, {}).get(basis.lower(), basis)


def _resolve_solvent_name(solvent: Optional[str], backend: str) -> Optional[str]:
    if solvent is None:
        return None
    key = solvent.strip().lower()
    return SOLVENT_ALIASES.get(key, {}).get(backend, solvent)


def _resolve_xtb_solvent_name(solvent: Optional[str]) -> Optional[str]:
    if solvent is None:
        return None
    return XTB_SOLVENT_ALIASES.get(solvent.strip().lower(), solvent)


def _default_plot_range(nucleus: str) -> Tuple[float, float]:
    return DEFAULT_PLOT_RANGES.get(nucleus, (-50.0, 250.0))


def _default_linewidth(nucleus: str) -> float:
    return DEFAULT_LINEWIDTHS.get(nucleus, 0.2)


def _resolve_references(
    nuclei: Sequence[str], user_refs: Dict[str, float]
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    warnings = []
    refs: Dict[str, Optional[float]] = {}
    for nucleus in nuclei:
        if nucleus in user_refs:
            refs[nucleus] = float(user_refs[nucleus])
            continue
        refs[nucleus] = DEFAULT_REFERENCE_SHIELDINGS.get(nucleus)
        if refs[nucleus] is not None:
            warnings.append(
                f"Using approximate built-in reference shielding for {nucleus} "
                f"({refs[nucleus]:.2f} ppm). Provide a calibrated reference for production use."
            )
        else:
            warnings.append(
                f"No built-in reference shielding is available for {nucleus}. "
                "Chemical shifts will be omitted unless you supply one."
            )
    return refs, warnings


def _command_for_backend(backend: str, command: Optional[str]) -> str:
    if command:
        return command
    defaults = {
        "orca": "orca",
        "nwchem": "nwchem",
        "gaussian": "g16",
    }
    return defaults[backend]


def _optimization_backend(settings: NMRSettings) -> str:
    return settings.optimization_backend or settings.backend


def _element_map_from_nuclei(nuclei: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for nucleus in nuclei:
        _, element = _parse_nucleus_label(nucleus)
        if element in mapping and mapping[element] != nucleus:
            raise ValueError(
                f"Multiple nuclei for element {element} are not supported in one run."
            )
        mapping[element] = nucleus
    return mapping


def _atoms_to_rdkit_mol(atoms: Atoms, charge: int = 0):
    if Chem is None:
        return None

    xyz = atoms2xyz(atoms)
    try:
        raw = Chem.MolFromXYZBlock(xyz)
        if raw is None:
            return None
        mol = Chem.Mol(raw)
        if rdDetermineBonds is not None:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        return mol
    except Exception:
        return None


def _should_sample_conformers(settings: NMRSettings, atoms: Atoms) -> bool:
    if settings.conformer_sampling is not None:
        return settings.conformer_sampling
    if Chem is None or rdMolDescriptors is None:
        return False
    mol = _atoms_to_rdkit_mol(atoms, charge=settings.charge)
    if mol is None:
        return False
    try:
        return rdMolDescriptors.CalcNumRotatableBonds(mol) > 0
    except Exception:
        return False


def _rdkit_conf_to_atoms(mol, conf_id: int) -> Atoms:
    conf = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = []
    for atom_index in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_index)
        positions.append((pos.x, pos.y, pos.z))
    return Atoms(symbols=symbols, positions=positions, pbc=False)


def generate_conformer_candidates(
    atoms: Atoms,
    settings: NMRSettings,
) -> Tuple[List[ConformerCandidate], List[str]]:
    """Generate conformer candidates, falling back to the input geometry."""

    warnings: List[str] = []
    if not _should_sample_conformers(settings, atoms):
        return [ConformerCandidate(conformer_id=0, atoms=atoms.copy())], warnings

    if Chem is None or AllChem is None:
        warnings.append(
            "RDKit is unavailable; falling back to the provided geometry without conformer sampling."
        )
        return [ConformerCandidate(conformer_id=0, atoms=atoms.copy())], warnings

    mol = _atoms_to_rdkit_mol(atoms, charge=settings.charge)
    if mol is None:
        warnings.append(
            "Could not convert the input geometry to an RDKit molecule; conformer sampling was skipped."
        )
        return [ConformerCandidate(conformer_id=0, atoms=atoms.copy())], warnings

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.pruneRmsThresh = settings.rmsd_prune_threshold
    params.enforceChirality = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True

    num_to_embed = max(settings.num_conformers * 3, settings.num_conformers)
    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=num_to_embed, params=params)
        )
    except Exception as exc:
        warnings.append(
            f"RDKit conformer generation failed ({exc}); falling back to the input geometry."
        )
        return [ConformerCandidate(conformer_id=0, atoms=atoms.copy())], warnings

    if not conf_ids:
        warnings.append(
            "RDKit did not generate any conformers; falling back to the input geometry."
        )
        return [ConformerCandidate(conformer_id=0, atoms=atoms.copy())], warnings

    has_mmff = bool(AllChem.MMFFHasAllMoleculeParams(mol))
    candidates: List[ConformerCandidate] = []
    for conf_id in conf_ids:
        energy_eV = None
        energy_source = ""
        try:
            if has_mmff:
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                props = AllChem.MMFFGetMoleculeProperties(mol)
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                if ff is not None:
                    energy_eV = float(ff.CalcEnergy()) * KCAL_MOL_TO_EV
                    energy_source = "mmff"
            else:
                AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                if ff is not None:
                    energy_eV = float(ff.CalcEnergy()) * KCAL_MOL_TO_EV
                    energy_source = "uff"
        except Exception:
            energy_eV = None
            energy_source = ""

        candidates.append(
            ConformerCandidate(
                conformer_id=int(conf_id),
                atoms=_rdkit_conf_to_atoms(mol, conf_id),
                initial_energy_eV=energy_eV,
                energy_source=energy_source,
            )
        )

    finite_energies = [
        c.initial_energy_eV for c in candidates if c.initial_energy_eV is not None
    ]
    if finite_energies:
        min_energy = min(finite_energies)
        filtered = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                float("inf")
                if item.initial_energy_eV is None
                else item.initial_energy_eV
            ),
        ):
            if candidate.initial_energy_eV is None:
                continue
            rel_kcal = (candidate.initial_energy_eV - min_energy) / KCAL_MOL_TO_EV
            if rel_kcal <= settings.relative_energy_window_kcal:
                filtered.append(candidate)
        candidates = filtered or candidates

    candidates = candidates[: settings.num_conformers]
    if not candidates:
        warnings.append(
            "No conformers survived pruning; falling back to the provided geometry."
        )
        return [ConformerCandidate(conformer_id=0, atoms=atoms.copy())], warnings

    return candidates, warnings


def _build_orca_simpleinput(
    method: str,
    basis: str,
    solvent_model: Optional[str],
    solvent: Optional[str],
    purpose: str,
) -> str:
    tokens = [
        _resolve_method(method, "orca"),
        _resolve_basis(basis, "orca"),
        "TightSCF",
        "RIJCOSX",
        "DEF2/J",
    ]
    if purpose == "opt":
        tokens.append("ENGRAD")
    elif purpose == "nmr":
        tokens.append("NMR")

    solvent_model = (solvent_model or "").strip().lower()
    solvent = _resolve_solvent_name(solvent, "orca")
    if solvent_model and solvent:
        tokens.append(f"{solvent_model.upper()}({solvent})")

    return " ".join(tokens)


def _build_orca_blocks(
    nuclei: Sequence[str],
    solvent_model: Optional[str],
    solvent: Optional[str],
    calculator_kwargs: Dict[str, object],
    purpose: str,
) -> str:
    lines = []

    pal_block = calculator_kwargs.get("pal_block")
    if pal_block:
        lines.append(str(pal_block).strip())
    else:
        lines.append("%pal nprocs 1 end")

    element_map = _element_map_from_nuclei(nuclei)
    if purpose == "nmr" and element_map:
        lines.append("%eprnmr")
        for element in element_map:
            lines.append(f"  Nuclei = all {element} {{shift}}")
        lines.append("end")

    solvent_model = (solvent_model or "").strip().lower()
    solvent = _resolve_solvent_name(solvent, "orca")
    if solvent_model == "smd" and solvent:
        lines.append("%cpcm")
        lines.append("  smd true")
        lines.append(f'  SMDsolvent "{solvent}"')
        lines.append("end")

    extra_blocks = calculator_kwargs.get("extra_blocks")
    if extra_blocks:
        lines.append(str(extra_blocks).strip())

    return "\n".join(line for line in lines if line)


def _build_gaussian_scrf(
    solvent_model: Optional[str], solvent: Optional[str]
) -> Optional[str]:
    if not solvent_model or not solvent:
        return None
    solvent_model = solvent_model.lower().strip()
    solvent = _resolve_solvent_name(solvent, "gaussian")
    return f"{solvent_model},solvent={solvent}"


def build_backend_calculator(
    settings: NMRSettings,
    purpose: str,
    directory: Path,
    job_name: str,
    nuclei: Optional[Sequence[str]] = None,
):
    """Build a backend-specific ASE calculator."""

    backend = settings.backend
    method = settings.optimization_method if purpose == "opt" else settings.method
    basis = settings.optimization_basis if purpose == "opt" else settings.basis
    command = _command_for_backend(backend, settings.command)
    calculator_kwargs = dict(settings.calculator_kwargs)

    if backend == "orca":
        from ase.calculators.orca import ORCA, OrcaProfile

        profile = OrcaProfile(command=command)
        return ORCA(
            profile=profile,
            directory=str(directory),
            charge=settings.charge,
            mult=settings.multiplicity,
            orcasimpleinput=_build_orca_simpleinput(
                method=method,
                basis=basis,
                solvent_model=settings.solvent_model,
                solvent=settings.solvent,
                purpose=purpose,
            ),
            orcablocks=_build_orca_blocks(
                nuclei=nuclei or settings.nuclei,
                solvent_model=settings.solvent_model,
                solvent=settings.solvent,
                calculator_kwargs=calculator_kwargs,
                purpose=purpose,
            ),
        )

    if backend == "gaussian":
        from ase.calculators.gaussian import Gaussian

        kwargs = {
            "label": str(directory / job_name),
            "command": command,
            "method": _resolve_method(method, "gaussian"),
            "basis": _resolve_basis(basis, "gaussian"),
            "charge": settings.charge,
            "mult": settings.multiplicity,
        }
        scrf = _build_gaussian_scrf(settings.solvent_model, settings.solvent)
        if scrf:
            kwargs["scrf"] = scrf
        if purpose == "nmr":
            kwargs["nmr"] = "giao"
        kwargs.update(calculator_kwargs)
        return Gaussian(**kwargs)

    if backend == "nwchem":
        from ase.calculators.nwchem import NWChem

        calculator_kwargs.pop("atom_indices", None)
        dft_block = dict(calculator_kwargs.pop("dft", {}))
        dft_block.setdefault("xc", _resolve_method(method, "nwchem"))
        if settings.multiplicity > 1:
            dft_block.setdefault("odft", None)
            dft_block.setdefault("mult", settings.multiplicity)

        kwargs = {
            "label": str(directory / job_name),
            "theory": "dft",
            "basis": _resolve_basis(basis, "nwchem"),
            "charge": settings.charge,
            "dft": dft_block,
        }
        if settings.command:
            kwargs["command"] = command

        solvent_model = (settings.solvent_model or "").strip().lower()
        if solvent_model and settings.solvent:
            cosmo_block = dict(calculator_kwargs.pop("cosmo", {}))
            if solvent_model == "smd":
                cosmo_block.setdefault("do_cosmo_smd", True)
                cosmo_block.setdefault(
                    "solvent", _resolve_solvent_name(settings.solvent, "nwchem")
                )
            kwargs["cosmo"] = cosmo_block

        if purpose == "nmr":
            kwargs["task"] = "property"
            kwargs["property"] = {"shielding": None}

        kwargs.update(calculator_kwargs)
        return NWChem(**kwargs)

    raise ValueError(f"Unsupported backend '{backend}'.")


def _nmr_output_path(backend: str, calculator) -> Path:
    if backend == "orca":
        return Path(calculator.directory) / "orca.out"
    if backend == "gaussian":
        return Path(calculator.directory) / f"{calculator.prefix}.log"
    if backend == "nwchem":
        return Path(calculator.directory) / f"{calculator.prefix}.nwo"
    raise ValueError(f"Unsupported backend '{backend}'.")


def _split_orca_blocks(orcablocks: str) -> Tuple[List[str], List[str]]:
    """Split ORCA blocks into pre-coordinate and post-coordinate groups."""

    if not orcablocks:
        return [], []

    lines = [line.rstrip() for line in str(orcablocks).splitlines() if line.strip()]
    blocks: List[str] = []
    current: List[str] = []
    in_block = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if stripped.startswith("%"):
            if current:
                blocks.append("\n".join(current))
                current = []
            current = [line]
            in_block = True
            if lower.endswith("end"):
                blocks.append("\n".join(current))
                current = []
                in_block = False
            continue
        if in_block:
            current.append(line)
            if lower == "end":
                blocks.append("\n".join(current))
                current = []
                in_block = False
        else:
            blocks.append(line)

    if current:
        blocks.append("\n".join(current))

    pre_coords: List[str] = []
    post_coords: List[str] = []
    for block in blocks:
        # ORCA reads %eprnmr after coordinates; before *xyz it can fail with
        # "nuclear properties are requested but no coordinates have been read".
        if block.lstrip().lower().startswith("%eprnmr"):
            post_coords.append(block)
        else:
            pre_coords.append(block)
    return pre_coords, post_coords


def _write_orca_input(path: Path, atoms: Atoms, parameters: Dict[str, object]) -> None:
    """Write ORCA input with the EPR/NMR block after the coordinates."""

    simpleinput = str(parameters.get("orcasimpleinput", "")).strip()
    charge = int(parameters.get("charge", 0))
    mult = int(parameters.get("mult", 1))
    pre_blocks, post_blocks = _split_orca_blocks(str(parameters.get("orcablocks", "")))

    lines = [f"! {simpleinput}"]
    lines.extend(pre_blocks)
    lines.append(f"*xyz {charge} {mult}")
    for atom in atoms:
        symbol = atom.symbol + " : " if atom.tag == 71 else atom.symbol
        lines.append(
            f"{symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}"
        )
    lines.append("*")
    lines.extend(post_blocks)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_nmr_job(calculator, atoms: Atoms, backend: str) -> Path:
    if backend == "orca":
        directory = Path(calculator.directory)
        directory.mkdir(parents=True, exist_ok=True)
        input_path = directory / "orca.inp"
        _write_orca_input(input_path, atoms, dict(calculator.parameters))
        calculator.template.execute(directory, calculator.profile)
    else:
        calculator.write_input(atoms, properties=["energy"], system_changes=all_changes)
        calculator.execute()
    return _nmr_output_path(backend, calculator)


def _parse_orca_energy(text: str) -> Optional[float]:
    match = re.search(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    if match:
        return float(match.group(1)) * Hartree
    return None


def parse_orca_nmr_output(
    path: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    in_table = False
    header_seen = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "CHEMICAL SHIELDING SUMMARY (ppm)" in raw_line:
            header_seen = True
            continue
        if not header_seen:
            continue
        if raw_line.strip().startswith("Nucleus"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line or set(line) <= {"-"}:
            continue
        match = re.match(
            r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
            raw_line,
        )
        if match:
            rows.append(
                {
                    "atom_index": int(match.group(1)),
                    "element": _normalize_element(match.group(2)),
                    "isotropic_shielding_ppm": float(match.group(3)),
                    "anisotropy_ppm": float(match.group(4)),
                }
            )
        elif rows:
            break
    if not rows:
        raise ValueError(f"Could not parse ORCA NMR shieldings from {path}.")
    meta = {"energy_eV": _parse_orca_energy(text)}
    return rows, meta


def parse_gaussian_nmr_output(
    path: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    pattern = re.compile(
        r"^\s*(\d+)\s+([A-Za-z]{1,2})\s+Isotropic\s*=\s*(-?\d+(?:\.\d+)?)\s+"
        r"Anisotropy\s*=\s*(-?\d+(?:\.\d+)?)",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "atom_index": int(match.group(1)) - 1,
                "element": _normalize_element(match.group(2)),
                "isotropic_shielding_ppm": float(match.group(3)),
                "anisotropy_ppm": float(match.group(4)),
            }
        )
    if not rows:
        raise ValueError(f"Could not parse Gaussian NMR shieldings from {path}.")
    return rows, {"energy_eV": None}


def parse_nwchem_nmr_output(
    path: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    atom_iter = list(re.finditer(r"Atom:\s+(\d+)\s+([A-Za-z]{1,2})", text))
    for index, match in enumerate(atom_iter):
        start = match.end()
        end = atom_iter[index + 1].start() if index + 1 < len(atom_iter) else len(text)
        block = text[start:end]
        iso_match = re.search(
            r"isotropic\s*=\s*(-?\d+(?:\.\d+)?)", block, re.IGNORECASE
        )
        aniso_match = re.search(
            r"anisotropy\s*=\s*(-?\d+(?:\.\d+)?)", block, re.IGNORECASE
        )
        if iso_match:
            rows.append(
                {
                    "atom_index": int(match.group(1)) - 1,
                    "element": _normalize_element(match.group(2)),
                    "isotropic_shielding_ppm": float(iso_match.group(1)),
                    "anisotropy_ppm": (
                        float(aniso_match.group(1)) if aniso_match else None
                    ),
                }
            )
    if not rows:
        raise ValueError(f"Could not parse NWChem NMR shieldings from {path}.")
    return rows, {"energy_eV": None}


def parse_nmr_output(
    backend: str, path: Path
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if backend == "orca":
        return parse_orca_nmr_output(path)
    if backend == "gaussian":
        return parse_gaussian_nmr_output(path)
    if backend == "nwchem":
        return parse_nwchem_nmr_output(path)
    raise ValueError(f"Unsupported backend '{backend}'.")


def _compute_boltzmann_weights(
    conformers: Sequence[ConformerCandidate],
    temperature: float,
) -> np.ndarray:
    energies = np.array(
        [
            (
                candidate.optimization_energy_eV
                if candidate.optimization_energy_eV is not None
                else (
                    candidate.initial_energy_eV
                    if candidate.initial_energy_eV is not None
                    else np.nan
                )
            )
            for candidate in conformers
        ],
        dtype=float,
    )
    finite_mask = np.isfinite(energies)
    if not finite_mask.any():
        return np.ones(len(conformers), dtype=float) / max(len(conformers), 1)

    finite_energies = energies[finite_mask]
    rel = finite_energies - finite_energies.min()
    beta = 1.0 / (K_BOLTZMANN_EV_PER_K * temperature)
    weights = np.exp(-beta * rel)
    weights /= weights.sum()

    all_weights = np.zeros(len(conformers), dtype=float)
    all_weights[finite_mask] = weights
    if (~finite_mask).any():
        remainder = 1.0 - all_weights.sum()
        if remainder > 0:
            all_weights[~finite_mask] = remainder / (~finite_mask).sum()
    return all_weights


def _lorentzian(x: np.ndarray, center: float, linewidth: float) -> np.ndarray:
    gamma = max(float(linewidth), 1e-8)
    half_gamma = gamma / 2.0
    return (half_gamma / np.pi) / ((x - center) ** 2 + half_gamma**2)


def _gaussian(x: np.ndarray, center: float, linewidth: float) -> np.ndarray:
    sigma = max(float(linewidth), 1e-8) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) / (
        sigma * math.sqrt(2.0 * math.pi)
    )


def _line_profile(
    x: np.ndarray,
    center: float,
    linewidth: float,
    lineshape: str,
    voigt_eta: float,
) -> np.ndarray:
    if lineshape == "lorentzian":
        return _lorentzian(x, center, linewidth)
    if lineshape == "gaussian":
        return _gaussian(x, center, linewidth)
    eta = min(max(float(voigt_eta), 0.0), 1.0)
    return eta * _lorentzian(x, center, linewidth) + (1.0 - eta) * _gaussian(
        x, center, linewidth
    )


def simulate_nmr_spectrum(
    peak_rows: Sequence[Dict[str, object]],
    nucleus: str,
    linewidth: float,
    plot_range: Tuple[float, float],
    lineshape: str,
    voigt_eta: float = 0.5,
    num_points: int = DEFAULT_GRID_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a one-dimensional NMR spectrum."""

    xmin, xmax = plot_range
    grid = np.linspace(min(xmin, xmax), max(xmin, xmax), num_points)
    intensity = np.zeros_like(grid)
    for row in peak_rows:
        if row.get("nucleus") != nucleus:
            continue
        shift = row.get("chemical_shift_ppm")
        if shift is None:
            continue
        amp = float(row.get("peak_intensity", 1.0))
        intensity += amp * _line_profile(
            grid, float(shift), linewidth, lineshape=lineshape, voigt_eta=voigt_eta
        )
    return grid, intensity


def _write_csv(
    path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _configure_matplotlib(output_dir: Path) -> None:
    mplconfigdir = output_dir / ".matplotlib"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))


def _plot_spectra(
    spectra: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    _configure_matplotlib(output_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nuclei = list(spectra)
    fig, axes = plt.subplots(len(nuclei), 1, figsize=(8, 3.2 * max(len(nuclei), 1)))
    if len(nuclei) == 1:
        axes = [axes]

    for ax, nucleus in zip(axes, nuclei):
        ppm = spectra[nucleus]["ppm"]
        intensity = spectra[nucleus]["intensity"]
        plot_range = spectra[nucleus]["plot_range"]
        ax.plot(ppm, intensity, linewidth=1.25)
        ax.set_xlabel("Chemical Shift (ppm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"{nucleus} NMR")
        ax.set_xlim(max(plot_range), min(plot_range))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_spectrum_csv(path: Path, ppm: np.ndarray, intensity: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ppm", "intensity"])
        for x, y in zip(ppm, intensity):
            writer.writerow([f"{float(x):.8f}", f"{float(y):.12e}"])


def _write_interactive_spectrum_html(
    output_path: Path,
    unique_name: str,
    peak_rows: Sequence[Dict[str, object]],
    settings: NMRSettings,
) -> None:
    peak_payload: Dict[str, List[Dict[str, object]]] = {}
    default_linewidths: Dict[str, float] = {}
    default_plot_ranges: Dict[str, List[float]] = {}
    for nucleus in settings.nuclei:
        linewidth = float(settings.linewidth or _default_linewidth(nucleus))
        plot_range = settings.plot_range or _default_plot_range(nucleus)
        default_linewidths[nucleus] = linewidth
        default_plot_ranges[nucleus] = [float(plot_range[0]), float(plot_range[1])]
        rows = []
        for row in peak_rows:
            if row.get("nucleus") != nucleus or row.get("chemical_shift_ppm") is None:
                continue
            rows.append(
                {
                    "atom_index": int(row["atom_index"]),
                    "conformer_id": int(row["conformer_id"]),
                    "element": str(row["element"]),
                    "shift": float(row["chemical_shift_ppm"]),
                    "intensity": float(row.get("peak_intensity", 1.0)),
                }
            )
        peak_payload[nucleus] = rows

    payload = {
        "title": f"{unique_name} NMR Viewer",
        "nuclei": list(settings.nuclei),
        "defaultLinewidths": default_linewidths,
        "defaultPlotRanges": default_plot_ranges,
        "defaultLineshape": settings.lineshape,
        "defaultVoigtEta": float(settings.voigt_eta),
        "gridPoints": min(DEFAULT_GRID_POINTS, 2048),
        "peaks": peak_payload,
    }
    payload_text = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html_text = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IQC NMR Viewer</title>
  <style>
    :root {
      --bg: #f3f0e9;
      --panel: #fffdf8;
      --ink: #1f1f1f;
      --muted: #6a6259;
      --line: #d8d0c4;
      --accent: #0f766e;
      --accent-strong: #115e59;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      background: linear-gradient(180deg, #efe7db 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero, .panel {
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.06);
    }
    .hero {
      padding: 24px;
      margin-bottom: 18px;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(1.8rem, 4vw, 2.8rem);
      line-height: 1.05;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 60rem;
    }
    .panel {
      padding: 18px;
      margin-bottom: 18px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      align-items: end;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .field label {
      font-size: 0.9rem;
      color: var(--muted);
    }
    .field input, .field select, .button {
      min-height: 42px;
      border-radius: 10px;
      border: 1px solid var(--line);
      padding: 8px 12px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .button {
      cursor: pointer;
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    .button.secondary {
      background: white;
      color: var(--accent-strong);
      border-color: var(--accent);
    }
    .canvas-wrap {
      background: white;
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 12px;
      overflow-x: auto;
    }
    canvas {
      width: 100%;
      height: auto;
      display: block;
    }
    .status {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 0.95rem;
      background: white;
      border-radius: 12px;
      overflow: hidden;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid #ece5da;
      text-align: left;
    }
    th {
      background: #f8f4ed;
      color: var(--muted);
      font-weight: 600;
    }
    tbody tr:last-child td {
      border-bottom: none;
    }
    @media (max-width: 700px) {
      .page { padding: 14px; }
      .panel, .hero { padding: 14px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1 id="viewer-title">IQC NMR Viewer</h1>
      <p>Adjust the simulated spectrum in the browser, then export the current view as a PNG.</p>
    </section>

    <section class="panel">
      <div class="controls">
        <div class="field">
          <label for="nucleus-select">Nucleus</label>
          <select id="nucleus-select"></select>
        </div>
        <div class="field">
          <label for="linewidth-input">Linewidth (ppm)</label>
          <input id="linewidth-input" type="number" min="0.001" step="0.01">
        </div>
        <div class="field">
          <label for="lineshape-select">Lineshape</label>
          <select id="lineshape-select">
            <option value="lorentzian">Lorentzian</option>
            <option value="gaussian">Gaussian</option>
            <option value="pseudo-voigt">Pseudo-Voigt</option>
          </select>
        </div>
        <div class="field">
          <label for="voigt-eta-input">Voigt Eta</label>
          <input id="voigt-eta-input" type="number" min="0" max="1" step="0.05">
        </div>
        <div class="field">
          <label for="ppm-min-input">Plot Min (ppm)</label>
          <input id="ppm-min-input" type="number" step="0.1">
        </div>
        <div class="field">
          <label for="ppm-max-input">Plot Max (ppm)</label>
          <input id="ppm-max-input" type="number" step="0.1">
        </div>
        <div class="actions">
          <button id="reset-button" class="button secondary" type="button">Reset Defaults</button>
          <button id="export-button" class="button" type="button">Export PNG</button>
        </div>
      </div>
      <div class="status" id="status-line"></div>
    </section>

    <section class="panel">
      <div class="canvas-wrap">
        <canvas id="spectrum-canvas" width="1200" height="700"></canvas>
      </div>
    </section>

    <section class="panel">
      <h2 style="margin-top:0">Peak List</h2>
      <table>
        <thead>
          <tr>
            <th>Atom</th>
            <th>Element</th>
            <th>Shift (ppm)</th>
            <th>Intensity</th>
            <th>Conformer</th>
          </tr>
        </thead>
        <tbody id="peak-table-body"></tbody>
      </table>
    </section>
  </div>

  <script id="nmr-data" type="application/json">__PAYLOAD__</script>
  <script>
    const payload = JSON.parse(document.getElementById("nmr-data").textContent);
    const titleEl = document.getElementById("viewer-title");
    const nucleusSelect = document.getElementById("nucleus-select");
    const linewidthInput = document.getElementById("linewidth-input");
    const lineshapeSelect = document.getElementById("lineshape-select");
    const voigtEtaInput = document.getElementById("voigt-eta-input");
    const ppmMinInput = document.getElementById("ppm-min-input");
    const ppmMaxInput = document.getElementById("ppm-max-input");
    const statusLine = document.getElementById("status-line");
    const peakTableBody = document.getElementById("peak-table-body");
    const canvas = document.getElementById("spectrum-canvas");
    const context = canvas.getContext("2d");

    titleEl.textContent = payload.title;
    payload.nuclei.forEach((nucleus) => {
      const option = document.createElement("option");
      option.value = nucleus;
      option.textContent = nucleus;
      nucleusSelect.appendChild(option);
    });
    lineshapeSelect.value = payload.defaultLineshape;

    function safeSlug(value) {
      return String(value).replace(/[^a-zA-Z0-9._-]+/g, "_");
    }

    function getDefaultRange(nucleus) {
      const values = payload.defaultPlotRanges[nucleus] || [-1, 12];
      return [Number(values[0]), Number(values[1])];
    }

    function resetControls() {
      const nucleus = nucleusSelect.value;
      linewidthInput.value = payload.defaultLinewidths[nucleus];
      voigtEtaInput.value = payload.defaultVoigtEta;
      const range = getDefaultRange(nucleus);
      ppmMinInput.value = range[0];
      ppmMaxInput.value = range[1];
      lineshapeSelect.value = payload.defaultLineshape;
      render();
    }

    function lorentzian(x, center, linewidth) {
      const gamma = Math.max(Number(linewidth), 1e-8);
      const halfGamma = gamma / 2.0;
      return (halfGamma / Math.PI) / (((x - center) ** 2) + (halfGamma ** 2));
    }

    function gaussian(x, center, linewidth) {
      const sigma = Math.max(Number(linewidth), 1e-8) / (2.0 * Math.sqrt(2.0 * Math.log(2.0)));
      return Math.exp(-((x - center) ** 2) / (2.0 * sigma ** 2)) / (sigma * Math.sqrt(2.0 * Math.PI));
    }

    function lineProfile(x, center, linewidth, lineshape, eta) {
      if (lineshape === "lorentzian") {
        return lorentzian(x, center, linewidth);
      }
      if (lineshape === "gaussian") {
        return gaussian(x, center, linewidth);
      }
      const clippedEta = Math.min(Math.max(Number(eta), 0.0), 1.0);
      return clippedEta * lorentzian(x, center, linewidth) + (1.0 - clippedEta) * gaussian(x, center, linewidth);
    }

    function buildSpectrum(nucleus, linewidth, plotMin, plotMax, lineshape, eta) {
      const peaks = payload.peaks[nucleus] || [];
      const lo = Math.min(plotMin, plotMax);
      const hi = Math.max(plotMin, plotMax);
      const points = Math.max(256, Number(payload.gridPoints || 2048));
      const ppm = [];
      const intensity = [];
      for (let i = 0; i < points; i += 1) {
        const x = lo + ((hi - lo) * i) / (points - 1);
        let y = 0.0;
        for (const peak of peaks) {
          y += Number(peak.intensity) * lineProfile(x, Number(peak.shift), linewidth, lineshape, eta);
        }
        ppm.push(x);
        intensity.push(y);
      }
      return { ppm, intensity, lo, hi, peaks };
    }

    function renderPeakTable(peaks) {
      peakTableBody.innerHTML = "";
      const rows = [...peaks].sort((a, b) => Number(b.shift) - Number(a.shift));
      for (const peak of rows) {
        const tr = document.createElement("tr");
        tr.innerHTML = [
          `<td>${peak.atom_index}</td>`,
          `<td>${peak.element}</td>`,
          `<td>${Number(peak.shift).toFixed(3)}</td>`,
          `<td>${Number(peak.intensity).toFixed(3)}</td>`,
          `<td>${peak.conformer_id}</td>`,
        ].join("");
        peakTableBody.appendChild(tr);
      }
    }

    function drawSpectrum(spectrum, nucleus, linewidth, lineshape, eta) {
      const width = canvas.width;
      const height = canvas.height;
      const marginLeft = 90;
      const marginRight = 28;
      const marginTop = 40;
      const marginBottom = 70;
      const plotWidth = width - marginLeft - marginRight;
      const plotHeight = height - marginTop - marginBottom;
      const maxIntensity = Math.max(...spectrum.intensity, 1e-12);

      context.clearRect(0, 0, width, height);
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, width, height);

      context.strokeStyle = "#d8d0c4";
      context.lineWidth = 1;
      context.strokeRect(marginLeft, marginTop, plotWidth, plotHeight);

      context.fillStyle = "#1f1f1f";
      context.font = "600 28px Georgia, serif";
      context.fillText(`${nucleus} NMR`, marginLeft, 28);

      context.font = "16px Georgia, serif";
      context.fillStyle = "#6a6259";
      context.fillText(`linewidth ${Number(linewidth).toFixed(3)} ppm, ${lineshape}, eta ${Number(eta).toFixed(2)}`, marginLeft, height - 18);

      const ticks = 6;
      context.font = "14px Georgia, serif";
      for (let i = 0; i <= ticks; i += 1) {
        const fraction = i / ticks;
        const x = marginLeft + fraction * plotWidth;
        const ppmValue = spectrum.hi - fraction * (spectrum.hi - spectrum.lo);
        context.strokeStyle = "#ece5da";
        context.beginPath();
        context.moveTo(x, marginTop);
        context.lineTo(x, marginTop + plotHeight);
        context.stroke();
        context.fillStyle = "#6a6259";
        context.textAlign = "center";
        context.fillText(ppmValue.toFixed(1), x, marginTop + plotHeight + 22);
      }

      for (let i = 0; i <= 4; i += 1) {
        const fraction = i / 4;
        const y = marginTop + plotHeight - fraction * plotHeight;
        context.strokeStyle = "#ece5da";
        context.beginPath();
        context.moveTo(marginLeft, y);
        context.lineTo(marginLeft + plotWidth, y);
        context.stroke();
        context.fillStyle = "#6a6259";
        context.textAlign = "right";
        context.fillText((fraction * maxIntensity).toExponential(1), marginLeft - 8, y + 4);
      }

      context.strokeStyle = "#0f766e";
      context.lineWidth = 2.2;
      context.beginPath();
      for (let i = 0; i < spectrum.ppm.length; i += 1) {
        const x = marginLeft + ((spectrum.hi - spectrum.ppm[i]) / (spectrum.hi - spectrum.lo)) * plotWidth;
        const y = marginTop + plotHeight - (spectrum.intensity[i] / maxIntensity) * plotHeight;
        if (i === 0) {
          context.moveTo(x, y);
        } else {
          context.lineTo(x, y);
        }
      }
      context.stroke();

      context.save();
      context.translate(20, marginTop + plotHeight / 2);
      context.rotate(-Math.PI / 2);
      context.textAlign = "center";
      context.fillStyle = "#1f1f1f";
      context.fillText("Intensity (a.u.)", 0, 0);
      context.restore();

      context.textAlign = "center";
      context.fillStyle = "#1f1f1f";
      context.fillText("Chemical Shift (ppm)", marginLeft + plotWidth / 2, height - 18);
    }

    function render() {
      const nucleus = nucleusSelect.value;
      const linewidth = Math.max(Number(linewidthInput.value), 0.001);
      const plotMin = Number(ppmMinInput.value);
      const plotMax = Number(ppmMaxInput.value);
      const eta = Math.min(Math.max(Number(voigtEtaInput.value), 0.0), 1.0);
      const lineshape = lineshapeSelect.value;
      const spectrum = buildSpectrum(nucleus, linewidth, plotMin, plotMax, lineshape, eta);
      drawSpectrum(spectrum, nucleus, linewidth, lineshape, eta);
      renderPeakTable(spectrum.peaks);
      statusLine.textContent = `${spectrum.peaks.length} peaks rendered for ${nucleus}. The x-axis follows NMR convention with larger ppm values on the left.`;
    }

    document.getElementById("reset-button").addEventListener("click", resetControls);
    document.getElementById("export-button").addEventListener("click", () => {
      const nucleus = nucleusSelect.value;
      const link = document.createElement("a");
      link.download = `${safeSlug(payload.title)}_${safeSlug(nucleus)}.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    });

    [
      linewidthInput,
      lineshapeSelect,
      voigtEtaInput,
      ppmMinInput,
      ppmMaxInput,
    ].forEach((element) => {
      element.addEventListener("input", render);
      element.addEventListener("change", render);
    });
    nucleusSelect.addEventListener("change", resetControls);

    if (payload.nuclei.length > 0) {
      nucleusSelect.value = payload.nuclei[0];
      resetControls();
    }
  </script>
</body>
</html>
"""
    output_path.write_text(
        html_text.replace("__PAYLOAD__", payload_text),
        encoding="utf-8",
    )


def optimize_conformer(
    candidate: ConformerCandidate,
    settings: NMRSettings,
    unique_name: str,
) -> ConformerCandidate:
    """Optimize a conformer geometry with the selected backend."""

    output_dir = candidate.output_dir or Path(".")
    opt_dir = output_dir / "opt"
    opt_dir.mkdir(parents=True, exist_ok=True)
    optimization_backend = _optimization_backend(settings)

    if optimization_backend == "xtb":
        return optimize_conformer_with_xtb(candidate, settings, unique_name)

    atoms = candidate.atoms.copy()
    local_settings = NMRSettings(**asdict(settings))
    local_settings.backend = optimization_backend
    calculator = build_backend_calculator(
        settings=local_settings,
        purpose="opt",
        directory=opt_dir,
        job_name=f"{unique_name}_conf{candidate.conformer_id:02d}_opt",
    )
    atoms.calc = calculator

    trajectory_path = opt_dir / "opt.traj"
    optimizer = BFGS(atoms, trajectory=str(trajectory_path))
    converged = optimizer.run(
        fmax=settings.optimization_fmax, steps=settings.optimization_max_steps
    )
    candidate.atoms = atoms
    candidate.optimized_xyz = atoms2xyz(atoms)
    candidate.optimization_energy_eV = float(atoms.get_potential_energy())
    candidate.optimization_converged = bool(converged)
    candidate.energy_source = "optimization"
    write(opt_dir / "optimized.xyz", atoms, format="xyz")
    return candidate


def _xtb_method_args(method: str) -> List[str]:
    key = (method or "").strip().lower()
    if key in {"", "gfn2", "gfn2-xtb", "gfn2xtb"}:
        return ["--gfn", "2"]
    if key in {"gfn1", "gfn1-xtb", "gfn1xtb"}:
        return ["--gfn", "1"]
    if key in {"gfnff", "gfn-ff", "gfn_ff"}:
        return ["--gfnff"]
    raise ValueError(
        "Unsupported xTB optimization method. Use GFN2-xTB, GFN1-xTB, or GFN-FF."
    )


def _parse_xtb_total_energy_eV(log_text: str) -> Optional[float]:
    patterns = [
        r"TOTAL ENERGY\s+(-?\d+\.\d+)",
        r"\|\s*TOTAL ENERGY\s+(-?\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, log_text)
        if match:
            return float(match.group(1)) * Hartree
    return None


def _read_xtb_optimized_atoms(opt_dir: Path) -> Tuple[Atoms, str]:
    opt_xyz = opt_dir / "xtbopt.xyz"
    if opt_xyz.exists():
        return read(opt_xyz), "xtbopt.xyz"

    opt_log = opt_dir / "xtbopt.log"
    if opt_log.exists():
        try:
            return read(opt_log, format="xyz"), "xtbopt.log"
        except Exception as exc:  # pragma: no cover - defensive path
            raise RuntimeError(
                f"xTB produced xtbopt.log but it could not be parsed: {exc}"
            ) from exc

    raise RuntimeError(
        "xTB optimization finished without producing xtbopt.xyz or xtbopt.log."
    )


def optimize_conformer_with_xtb(
    candidate: ConformerCandidate,
    settings: NMRSettings,
    unique_name: str,
) -> ConformerCandidate:
    """Optimize a conformer using the xTB command-line program."""

    xtb_executable = shutil.which("xtb")
    if xtb_executable is None:
        raise RuntimeError(
            "xTB executable not found on PATH. Install xTB or choose a different optimization backend."
        )

    output_dir = candidate.output_dir or Path(".")
    opt_dir = output_dir / "opt"
    opt_dir.mkdir(parents=True, exist_ok=True)

    input_xyz = opt_dir / "input.xyz"
    write(input_xyz, candidate.atoms, format="xyz")

    cmd = [
        xtb_executable,
        input_xyz.name,
        "--opt",
        "normal",
        "--chrg",
        str(settings.charge),
        "--uhf",
        str(max(settings.multiplicity - 1, 0)),
        "--parallel",
        "1",
    ]
    cmd.extend(_xtb_method_args(settings.optimization_method))

    solvent_model = (settings.solvent_model or "").strip().lower()
    xtb_solvent = _resolve_xtb_solvent_name(settings.solvent)
    if xtb_solvent:
        if solvent_model in {"", "none"}:
            pass
        elif solvent_model in {"alpb", "smd"}:
            cmd.extend(["--alpb", xtb_solvent])
        elif solvent_model in {"gbsa", "cpcm", "cosmo"}:
            cmd.extend(["--gbsa", xtb_solvent])
        else:
            logging.warning(
                "Unsupported xTB solvent model '%s'; skipping solvent for xTB optimization.",
                settings.solvent_model,
            )

    log_path = opt_dir / "xtb.out"
    run_error = None
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            subprocess.run(
                cmd,
                cwd=opt_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            run_error = exc

    try:
        optimized_atoms, geometry_source = _read_xtb_optimized_atoms(opt_dir)
    except RuntimeError as exc:
        if run_error is not None:
            raise RuntimeError(
                f"xTB optimization failed without a recoverable geometry: {run_error}"
            ) from run_error
        raise exc

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    if run_error is not None:
        logging.warning(
            "xTB optimization returned a non-zero exit status for %s; using the latest geometry from %s.",
            unique_name,
            geometry_source,
        )

    candidate.atoms = optimized_atoms
    candidate.optimized_xyz = atoms2xyz(optimized_atoms)
    candidate.optimization_energy_eV = _parse_xtb_total_energy_eV(log_text)
    candidate.optimization_converged = run_error is None
    candidate.energy_source = "xtb"
    write(opt_dir / "optimized.xyz", optimized_atoms, format="xyz")
    return candidate


def run_nmr_workflow(
    atoms: Atoms,
    unique_name: str = "",
    **params,
) -> Tuple[Atoms, Dict[str, object]]:
    """Run a backend-aware NMR workflow and generate plots and tabulated outputs."""

    settings = build_nmr_settings(**params)
    if not unique_name:
        unique_name = "nmr"

    output_dir = Path(settings.output_dir or f"{unique_name}_nmr").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir = str(output_dir)

    results: Dict[str, object] = {
        "warnings": [],
        "error": "",
        "nmr_backend": settings.backend,
        "nmr_optimization_backend": _optimization_backend(settings),
        "nmr_nuclei": settings.nuclei,
        "nmr_method": settings.method,
        "nmr_basis": settings.basis,
        "nmr_solvent_model": settings.solvent_model,
        "nmr_solvent": settings.solvent,
        "nmr_output_dir": str(output_dir),
        "nmr_settings": asdict(settings),
    }

    references, reference_warnings = _resolve_references(
        settings.nuclei, settings.reference_shieldings
    )
    results["nmr_reference_shieldings_ppm"] = references
    results["warnings"].extend(reference_warnings)
    if _optimization_backend(settings) == "xtb" and settings.optimize_geometry:
        results["warnings"].append(
            "xTB pre-optimization ignores optimization_basis and uses the xTB Hamiltonian selected by optimization_method."
        )

    try:
        conformers, conformer_warnings = generate_conformer_candidates(atoms, settings)
    except Exception as exc:
        results["error"] = f"Failed to prepare conformers: {exc}"
        return atoms, results

    results["warnings"].extend(conformer_warnings)
    results["nmr_requested_num_conformers"] = settings.num_conformers
    results["nmr_conformer_sampling"] = len(conformers) > 1

    successful_conformers: List[ConformerCandidate] = []
    atom_rows: List[Dict[str, object]] = []
    element_to_nucleus = _element_map_from_nuclei(settings.nuclei)

    for index, candidate in enumerate(conformers):
        candidate.output_dir = output_dir / f"conf_{index:02d}"
        candidate.output_dir.mkdir(parents=True, exist_ok=True)
        local_atoms = candidate.atoms.copy()
        candidate.atoms = local_atoms

        try:
            if settings.optimize_geometry:
                candidate = optimize_conformer(candidate, settings, unique_name)
                if candidate.optimization_converged is False:
                    results["warnings"].append(
                        f"Geometry optimization did not fully converge for conformer {candidate.conformer_id}; using the latest available geometry."
                    )
            else:
                write(candidate.output_dir / "input.xyz", candidate.atoms, format="xyz")
        except Exception as exc:
            results["warnings"].append(
                f"Geometry optimization failed for conformer {candidate.conformer_id}: {exc}"
            )
            continue

        try:
            nmr_dir = candidate.output_dir / "nmr"
            nmr_dir.mkdir(parents=True, exist_ok=True)

            atom_indices = [
                atom_index + 1
                for atom_index, symbol in enumerate(
                    candidate.atoms.get_chemical_symbols()
                )
                if _normalize_element(symbol) in element_to_nucleus
            ]
            if not atom_indices:
                results["warnings"].append(
                    f"No atoms matched the requested nuclei for conformer {candidate.conformer_id}."
                )
                continue

            calculator_kwargs = dict(settings.calculator_kwargs)
            calculator_kwargs["atom_indices"] = atom_indices
            local_settings = NMRSettings(**asdict(settings))
            local_settings.calculator_kwargs = calculator_kwargs

            calculator = build_backend_calculator(
                settings=local_settings,
                purpose="nmr",
                directory=nmr_dir,
                job_name=f"{unique_name}_conf{candidate.conformer_id:02d}_nmr",
                nuclei=settings.nuclei,
            )
            output_path = _run_nmr_job(
                calculator=calculator,
                atoms=candidate.atoms.copy(),
                backend=settings.backend,
            )
            shielding_rows, meta = parse_nmr_output(settings.backend, output_path)
            if (
                meta.get("energy_eV") is not None
                and candidate.optimization_energy_eV is None
            ):
                candidate.optimization_energy_eV = float(meta["energy_eV"])
        except Exception as exc:
            results["warnings"].append(
                f"NMR calculation failed for conformer {candidate.conformer_id}: {exc}"
            )
            continue

        successful_conformers.append(candidate)
        energy_eV = (
            candidate.optimization_energy_eV
            if candidate.optimization_energy_eV is not None
            else candidate.initial_energy_eV
        )
        for row in shielding_rows:
            element = _normalize_element(str(row["element"]))
            nucleus = element_to_nucleus.get(element)
            if nucleus is None:
                continue
            reference = references.get(nucleus)
            shift = (
                None
                if reference is None
                else float(reference) - float(row["isotropic_shielding_ppm"])
            )
            atom_rows.append(
                {
                    "conformer_id": candidate.conformer_id,
                    "atom_index": int(row["atom_index"]),
                    "element": element,
                    "nucleus": nucleus,
                    "isotropic_shielding_ppm": float(row["isotropic_shielding_ppm"]),
                    "chemical_shift_ppm": shift,
                    "anisotropy_ppm": row.get("anisotropy_ppm"),
                    "conformer_energy_eV": energy_eV,
                    "conformer_energy_source": candidate.energy_source
                    or (
                        "optimization"
                        if candidate.optimization_energy_eV is not None
                        else ""
                    ),
                    "boltzmann_weight": None,
                    "peak_intensity": None,
                }
            )

    if not successful_conformers or not atom_rows:
        results["error"] = "No NMR results were produced."
        return atoms, results

    weights = _compute_boltzmann_weights(successful_conformers, settings.temperature)
    weight_map = {
        candidate.conformer_id: float(weight)
        for candidate, weight in zip(successful_conformers, weights)
    }
    for candidate, weight in zip(successful_conformers, weights):
        candidate.boltzmann_weight = float(weight)

    for row in atom_rows:
        row["boltzmann_weight"] = weight_map[row["conformer_id"]]
        row["peak_intensity"] = weight_map[row["conformer_id"]]

    weighted_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for row in atom_rows:
        grouped.setdefault((row["atom_index"], row["nucleus"]), []).append(row)

    for (atom_index, nucleus), rows in sorted(grouped.items()):
        total_weight = sum(float(row["boltzmann_weight"]) for row in rows)
        weighted_shielding = sum(
            float(row["isotropic_shielding_ppm"]) * float(row["boltzmann_weight"])
            for row in rows
        )
        weighted_shift = None
        if rows[0]["chemical_shift_ppm"] is not None:
            weighted_shift = sum(
                float(row["chemical_shift_ppm"]) * float(row["boltzmann_weight"])
                for row in rows
            )
        weighted_rows.append(
            {
                "atom_index": atom_index,
                "element": rows[0]["element"],
                "nucleus": nucleus,
                "weighted_isotropic_shielding_ppm": weighted_shielding,
                "weighted_chemical_shift_ppm": weighted_shift,
                "total_weight": total_weight,
            }
        )

    spectra: Dict[str, Dict[str, np.ndarray]] = {}
    spectrum_paths: Dict[str, str] = {}
    for nucleus in settings.nuclei:
        linewidth = settings.linewidth or _default_linewidth(nucleus)
        plot_range = settings.plot_range or _default_plot_range(nucleus)
        ppm, intensity = simulate_nmr_spectrum(
            peak_rows=atom_rows,
            nucleus=nucleus,
            linewidth=linewidth,
            plot_range=plot_range,
            lineshape=settings.lineshape,
            voigt_eta=settings.voigt_eta,
        )
        spectra[nucleus] = {
            "ppm": ppm,
            "intensity": intensity,
            "plot_range": np.array(plot_range, dtype=float),
        }
        spectrum_csv = output_dir / f"{unique_name}_{nucleus}_spectrum.csv"
        _write_spectrum_csv(spectrum_csv, ppm, intensity)
        spectrum_paths[nucleus] = str(spectrum_csv)

    plot_path = output_dir / f"{unique_name}_nmr_spectra.png"
    _plot_spectra(spectra, plot_path)
    interactive_html = output_dir / f"{unique_name}_nmr_viewer.html"
    _write_interactive_spectrum_html(
        output_path=interactive_html,
        unique_name=unique_name,
        peak_rows=atom_rows,
        settings=settings,
    )

    atom_results_csv = output_dir / f"{unique_name}_nmr_atom_results.csv"
    atom_results_json = output_dir / f"{unique_name}_nmr_atom_results.json"
    weighted_csv = output_dir / f"{unique_name}_nmr_weighted_peaks.csv"
    weighted_json = output_dir / f"{unique_name}_nmr_weighted_peaks.json"
    conformer_csv = output_dir / f"{unique_name}_nmr_conformers.csv"

    atom_fieldnames = [
        "conformer_id",
        "atom_index",
        "element",
        "nucleus",
        "isotropic_shielding_ppm",
        "chemical_shift_ppm",
        "anisotropy_ppm",
        "conformer_energy_eV",
        "conformer_energy_source",
        "boltzmann_weight",
        "peak_intensity",
    ]
    _write_csv(atom_results_csv, atom_rows, atom_fieldnames)
    _write_json(atom_results_json, atom_rows)

    weighted_fieldnames = [
        "atom_index",
        "element",
        "nucleus",
        "weighted_isotropic_shielding_ppm",
        "weighted_chemical_shift_ppm",
        "total_weight",
    ]
    _write_csv(weighted_csv, weighted_rows, weighted_fieldnames)
    _write_json(weighted_json, weighted_rows)

    conformer_rows = [
        {
            "conformer_id": candidate.conformer_id,
            "energy_eV": (
                candidate.optimization_energy_eV
                if candidate.optimization_energy_eV is not None
                else candidate.initial_energy_eV
            ),
            "energy_source": candidate.energy_source
            or ("optimization" if candidate.optimization_energy_eV is not None else ""),
            "boltzmann_weight": candidate.boltzmann_weight,
            "optimization_converged": candidate.optimization_converged,
            "output_dir": str(candidate.output_dir) if candidate.output_dir else "",
        }
        for candidate in successful_conformers
    ]
    _write_csv(
        conformer_csv,
        conformer_rows,
        [
            "conformer_id",
            "energy_eV",
            "energy_source",
            "boltzmann_weight",
            "optimization_converged",
            "output_dir",
        ],
    )

    results.update(
        {
            "nmr_num_conformers_used": len(successful_conformers),
            "nmr_atom_results_csv": str(atom_results_csv),
            "nmr_atom_results_json": str(atom_results_json),
            "nmr_weighted_peaks_csv": str(weighted_csv),
            "nmr_weighted_peaks_json": str(weighted_json),
            "nmr_conformer_summary_csv": str(conformer_csv),
            "nmr_spectrum_plot": str(plot_path),
            "nmr_interactive_html": str(interactive_html),
            "nmr_spectrum_csv_files": spectrum_paths,
            "nmr_atom_results": atom_rows,
            "nmr_weighted_peaks": weighted_rows,
            "nmr_conformers": conformer_rows,
        }
    )

    final_atoms = min(
        successful_conformers,
        key=lambda candidate: (
            float("inf")
            if candidate.optimization_energy_eV is None
            and candidate.initial_energy_eV is None
            else (
                candidate.optimization_energy_eV
                if candidate.optimization_energy_eV is not None
                else candidate.initial_energy_eV
            )
        ),
    ).atoms
    return final_atoms, results
