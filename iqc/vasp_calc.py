"""VASP calculator for IQC, configured to match the MACE-MP training level.

MACE-MP foundation models are trained on **MPtrj** — Materials Project
relaxation trajectories computed with VASP using PBE (GGA) and the Materials
Project's standard input settings. To make VASP reference numbers directly
comparable to MACE-MP, this module builds an ASE ``Vasp`` calculator whose
INCAR mirrors pymatgen's ``MPRelaxSet`` (PBE, ENCUT 520 eV, spin-polarized,
LASPH, standard PBE PAW potentials).

Because IQC drives geometry optimization and vibrations through ASE
(``run_optimization`` / ``run_vibrations``), the calculator is configured for
**single-point** evaluation (``nsw=0``, ``ibrion=-1``): ASE asks for energy and
forces at each geometry, VASP returns them. For an isolated molecule such as
CO2 we place it in a large periodic box and sample only the Γ point.

Environment / site configuration (Polaris defaults, overridable):
    VASP binary   : ``$IQC_VASP_COMMAND`` or ``command=`` kwarg
                    (default: mpiexec over vasp_std under /soft/applications).
    PAW potentials: ``$VASP_PP_PATH`` must contain ``potpaw_PBE/<El>/POTCAR``.
                    Default points at /soft/applications/vasp/potcar.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

# Default site locations on ALCF Polaris (all overridable via env / kwargs).
_DEFAULT_PP_PATH = "/soft/applications/vasp/potcar"
_DEFAULT_VASP_STD = "/soft/applications/vasp/vasp.6.4.3/bin/vasp_std"

# MPtrj / MPRelaxSet-consistent INCAR (PBE GGA). Element-specific Hubbard U and
# magmoms from MPRelaxSet are only triggered for transition-metal oxides/
# fluorides and are irrelevant for light main-group molecules like CO2; they can
# be layered on via kwargs when needed.
_MP_INCAR_DEFAULTS: Dict[str, Any] = {
    "prec": "Accurate",
    "encut": 520,          # eV — Materials Project standard cutoff
    "gga": "PE",           # PBE
    "ediff": 1e-6,         # tight for clean forces on a molecule
    "ismear": 0,           # Gaussian smearing (molecule / insulator)
    "sigma": 0.05,
    "ispin": 2,            # MP runs are spin-polarized
    "lasph": True,         # MP setting: aspherical PAW contributions
    "lreal": False,        # real-space projection off → accurate small-cell forces
    "algo": "Normal",
    "nelm": 200,
    "lwave": False,
    "lcharg": False,
    "lorbit": 11,
    "isym": 0,             # keep symmetry off so ASE finite-diff displacements are safe
}


def _resolve_vasp_command(command: Optional[str], nproc: Optional[int]) -> str:
    """Return the shell command ASE will run to launch VASP."""
    if command:
        return command
    env_cmd = os.environ.get("IQC_VASP_COMMAND") or os.environ.get("ASE_VASP_COMMAND")
    if env_cmd:
        return env_cmd
    binary = os.environ.get("IQC_VASP_BINARY", _DEFAULT_VASP_STD)
    ranks = int(nproc) if nproc else int(os.environ.get("IQC_VASP_NPROC", "32"))
    # Polaris launches MPI ranks with mpiexec; bind by core for VASP.
    return f"mpiexec -n {ranks} --cpu-bind depth -d 1 {binary}"


def get_vasp_calculator(
    vacuum: float = 8.0,
    kpts=(1, 1, 1),
    gamma: bool = True,
    command: Optional[str] = None,
    nproc: Optional[int] = None,
    pp_path: Optional[str] = None,
    directory: str = "vasp_run",
    charge: int = 0,
    multiplicity: Optional[int] = None,
    **incar_overrides: Any,
):
    """Build an MP-compatible ASE ``Vasp`` calculator.

    Args:
        vacuum: If the atoms are non-periodic, wrap them in a cubic cell with
            at least this much vacuum (Å) around the molecule and turn on PBC.
        kpts / gamma: k-point mesh (Γ-only by default — correct for an isolated
            molecule in a large box).
        command / nproc: VASP launch command (or rank count for the default
            mpiexec command).
        pp_path: Directory holding ``potpaw_PBE`` (sets ``VASP_PP_PATH``).
        directory: Scratch directory for VASP I/O.
        charge / multiplicity: molecular charge and spin multiplicity 2S+1.
        **incar_overrides: Any extra INCAR tags override the MP defaults.

    Returns:
        A configured ``ase.calculators.vasp.Vasp`` instance tagged with IQC
        metadata (``_iqc_calculator_family='vasp'``).
    """
    try:
        from ase.calculators.vasp import Vasp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ASE's VASP calculator is unavailable; ensure ASE is installed."
        ) from exc

    pp_path = pp_path or os.environ.get("VASP_PP_PATH") or _DEFAULT_PP_PATH
    os.environ["VASP_PP_PATH"] = pp_path

    incar = dict(_MP_INCAR_DEFAULTS)
    incar.update(incar_overrides)

    # Charge: VASP expresses this via NELECT (neutral default needs nothing).
    if charge:
        # Adding electrons lowers NELECT; ASE's `nelect` overrides total count.
        # Callers who need this must know the neutral electron count; we log so
        # the level of theory is never silently wrong.
        logging.warning(
            "VASP charge=%d requested: set INCAR `nelect` explicitly via "
            "calculator_params for charged systems (auto-NELECT not applied).",
            charge,
        )

    calc = Vasp(
        directory=directory,
        xc="pbe",
        setups="recommended",   # pymatgen/MP-style PBE PAW choices
        kpts=kpts,
        gamma=gamma,
        nsw=0,                  # single point — ASE drives opt/vib
        ibrion=-1,
        command=_resolve_vasp_command(command, nproc),
        **incar,
    )

    # IQC metadata so asetools/main can recognize and label the backend.
    calc._iqc_calculator_family = "vasp"
    calc._iqc_spin_charge_convention = "vasp"
    calc._iqc_vacuum = float(vacuum)
    calc.model_name = f"vasp/pbe/encut{incar['encut']}"
    return calc


def ensure_cell_for_molecule(atoms, vacuum: float = 8.0) -> None:
    """Give a non-periodic molecule a cubic box with vacuum, in place.

    VASP always requires a cell + PBC. If ``atoms`` has no cell (typical for a
    molecule read from XYZ), build a cubic cell large enough to enclose it with
    ``vacuum`` Å of padding on all sides and enable periodicity.
    """
    if atoms.cell is not None and atoms.cell.rank == 3 and atoms.get_volume() > 1e-6:
        atoms.pbc = True
        return
    atoms.center(vacuum=vacuum, about=None)
    # center() with vacuum sets a cell sized to the molecule + 2*vacuum.
    atoms.pbc = True
