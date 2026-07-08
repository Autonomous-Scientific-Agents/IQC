"""ASE calculator for PySCF.

PySCF (https://pyscf.org) is a Python-native electronic-structure package.
This module wraps it behind the ASE ``Calculator`` interface so IQC can drive
it through the same ``get_potential_energy`` / ``get_forces`` conventions used
for every other backend.

Two families of methods are supported:

* **DFT / HF** (``method='dft'`` with an ``xc`` functional, or ``method='hf'``):
  energy, analytic **forces**, and **dipole** are all available. This is the
  workhorse for geometry optimization + vibrations + thermochemistry entirely
  at the PySCF level. To match the MACE-MP (MPtrj) training level of theory the
  default functional is **PBE**.

* **Correlated wavefunction methods** (``method`` in
  ``{'mp2','ccsd','ccsd(t)'}``): energy only. These have no cheap analytic
  gradient here, so they are meant to be used as the *energy* backend in a
  composite thermochemistry workflow (geometry + Hessian from a force-capable
  calculator such as MACE, single-point correlation energy from PySCF).

Unit conventions (ASE ⇄ PySCF):
    energy   : Hartree      → eV            (``ase.units.Hartree``)
    forces   : Hartree/Bohr → eV/Å          (``Hartree / Bohr``)
    dipole   : Debye        → e·Å           (``ase.units.Debye``)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from ase import units
from ase.calculators.calculator import (
    Calculator,
    CalculationFailed,
    all_changes,
)

_HARTREE_TO_EV = units.Hartree
_HARTREE_PER_BOHR_TO_EV_PER_ANG = units.Hartree / units.Bohr
_DEBYE_TO_E_ANG = units.Debye  # value of 1 Debye in ASE units (e·Å)

# User-facing method aliases → canonical names.
_METHOD_ALIASES: Dict[str, str] = {
    "dft": "dft",
    "ks": "dft",
    "rks": "dft",
    "uks": "dft",
    "hf": "hf",
    "scf": "hf",
    "rhf": "hf",
    "uhf": "hf",
    "mp2": "mp2",
    "ccsd": "ccsd",
    "ccsd(t)": "ccsd(t)",
    "ccsd_t": "ccsd(t)",
    "ccsd-t": "ccsd(t)",
}

_FORCE_METHODS = {"dft", "hf"}  # methods with analytic gradients wired here
_CORR_METHODS = {"mp2", "ccsd", "ccsd(t)"}


def _normalize_method(method: str) -> str:
    key = str(method).lower().replace(" ", "")
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unsupported PySCF method {method!r}. "
            f"Choose one of: {sorted(set(_METHOD_ALIASES))}"
        )
    return _METHOD_ALIASES[key]


class PySCFCalculator(Calculator):
    """Drive PySCF from ASE.

    Standard parameters:
        charge (int): Molecular charge. Default 0.
        multiplicity (int): Spin multiplicity 2S+1. If given it sets the number
            of unpaired electrons (``spin = multiplicity - 1``). ``spin`` may be
            passed directly to override.

    PySCF-specific parameters:
        method (str): 'dft' (default), 'hf', 'mp2', 'ccsd', 'ccsd(t)'.
        xc (str): Exchange-correlation functional for ``method='dft'``.
            Default 'pbe' (matches the MACE-MP / MPtrj training level).
        basis (str): Basis set. Default 'def2-svp'.
        spin (int|None): 2S = n_alpha - n_beta (number of unpaired electrons).
            Derived from ``multiplicity`` when None.
        conv_tol (float): SCF energy convergence. Default 1e-9.
        max_cycle (int): SCF max iterations. Default 200.
        density_fit (bool): Use density fitting (RI) for the SCF. Default False.
        frozen (int|None): Frozen core orbitals for correlated methods.
        verbose (int): PySCF verbosity. Default 0.
    """

    implemented_properties = ["energy", "free_energy", "forces", "dipole"]

    default_parameters: Dict[str, Any] = {
        "method": "dft",
        "xc": "pbe",
        "basis": "def2-svp",
        "charge": 0,
        "multiplicity": None,
        "spin": None,
        "conv_tol": 1e-9,
        "max_cycle": 200,
        "density_fit": False,
        "frozen": None,
        "verbose": 0,
    }

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=Calculator._deprecated,
        label: str = "pyscf",
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
        self._iqc_calculator_family = "pyscf"
        self._iqc_spin_charge_convention = "pyscf"
        # Human-facing label used by the orchestrator's result metadata.
        method = _normalize_method(self.parameters["method"])
        if method == "dft":
            self.model_name = f"pyscf/{self.parameters['xc']}/{self.parameters['basis']}"
        else:
            self.model_name = f"pyscf/{method}/{self.parameters['basis']}"
        # Only DFT/HF have analytic forces + dipole wired here. Correlated
        # methods (MP2/CCSD/CCSD(T)) are energy-only; advertising that lets IQC
        # add finite-difference forces via NumericalForceCalculator.
        if method in _FORCE_METHODS:
            self.implemented_properties = ["energy", "free_energy", "forces", "dipole"]
        else:
            self.implemented_properties = ["energy", "free_energy"]

    # ------------------------------------------------------------------
    def _build_mol(self):
        from pyscf import gto

        params = self.parameters
        atoms = self.atoms
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        spin = params.get("spin")
        if spin is None:
            mult = params.get("multiplicity")
            spin = 0 if mult is None else int(mult) - 1

        mol = gto.Mole()
        mol.atom = [
            (sym, (float(p[0]), float(p[1]), float(p[2])))
            for sym, p in zip(symbols, positions)
        ]
        mol.unit = "Angstrom"
        mol.basis = params["basis"]
        mol.charge = int(params["charge"])
        mol.spin = int(spin)
        mol.verbose = int(params["verbose"])
        mol.build()
        return mol, int(spin)

    def _make_scf(self, mol, spin):
        """Return an (unconverged) mean-field object appropriate for the spin."""
        from pyscf import dft, scf

        params = self.parameters
        method = _normalize_method(params["method"])
        unrestricted = spin != 0

        if method == "dft":
            mf = dft.UKS(mol) if unrestricted else dft.RKS(mol)
            mf.xc = params["xc"]
        else:
            # HF reference (also used as the reference for correlated methods).
            mf = scf.UHF(mol) if unrestricted else scf.RHF(mol)

        if params.get("density_fit"):
            mf = mf.density_fit()
        mf.conv_tol = float(params["conv_tol"])
        mf.max_cycle = int(params["max_cycle"])
        mf.verbose = int(params["verbose"])
        return mf

    # ------------------------------------------------------------------
    def calculate(
        self,
        atoms=None,
        properties: Sequence[str] = ("energy",),
        system_changes: Iterable[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.atoms is None:
            raise CalculationFailed("PySCFCalculator: no atoms attached")

        method = _normalize_method(self.parameters["method"])
        try:
            mol, spin = self._build_mol()
            if method in ("dft", "hf"):
                self._run_scf(mol, spin, properties)
            else:
                self._run_correlated(mol, spin, method)
        except CalculationFailed:
            raise
        except Exception as exc:  # pragma: no cover - surfaced to ASE
            raise CalculationFailed(f"PySCF {method} calculation failed: {exc}") from exc

    def _run_scf(self, mol, spin, properties):
        t0 = time.time()
        mf = self._make_scf(mol, spin)
        e_hartree = mf.kernel()
        if not mf.converged:
            logging.warning("PySCF SCF did not converge (method=%s)", self.parameters["method"])
        self.results["energy"] = e_hartree * _HARTREE_TO_EV
        self.results["free_energy"] = self.results["energy"]
        self.results["scf_energy_eV"] = e_hartree * _HARTREE_TO_EV
        self.results["total_energy_eV"] = e_hartree * _HARTREE_TO_EV
        self.results["scf_time_s"] = time.time() - t0

        # Analytic forces (always cheap for HF/DFT; compute so opt/vib work).
        try:
            grad = mf.nuc_grad_method().kernel()  # Hartree/Bohr, dE/dR
            self.results["forces"] = -np.asarray(grad) * _HARTREE_PER_BOHR_TO_EV_PER_ANG
        except Exception as exc:
            logging.warning("PySCF gradient unavailable: %s", exc)

        # Dipole moment (Debye → e·Å).
        try:
            dip_debye = np.asarray(mf.dip_moment(verbose=0))
            self.results["dipole"] = dip_debye * _DEBYE_TO_E_ANG
        except Exception as exc:
            logging.warning("PySCF dipole unavailable: %s", exc)

    def _run_correlated(self, mol, spin, method):
        from pyscf import cc, mp

        params = self.parameters
        frozen = params.get("frozen")

        # HF reference.
        t0 = time.time()
        mf = self._make_scf(mol, spin)  # method != dft/hf ⇒ RHF/UHF reference
        # Force an HF reference regardless of the 'method' string.
        from pyscf import scf

        if not isinstance(mf, (scf.uhf.UHF, scf.rhf.RHF)):
            mf = scf.UHF(mol) if spin != 0 else scf.RHF(mol)
            mf.conv_tol = float(params["conv_tol"])
            mf.max_cycle = int(params["max_cycle"])
            mf.verbose = int(params["verbose"])
        e_scf = mf.kernel()
        if not mf.converged:
            logging.warning("PySCF HF reference did not converge")
        scf_time = time.time() - t0
        self.results["scf_energy_eV"] = e_scf * _HARTREE_TO_EV
        self.results["scf_time_s"] = scf_time

        total_hartree = e_scf

        if method == "mp2":
            t1 = time.time()
            mp2 = mp.MP2(mf, frozen=frozen) if frozen else mp.MP2(mf)
            e_corr, _ = mp2.kernel()
            self.results["mp2_correlation_eV"] = e_corr * _HARTREE_TO_EV
            self.results["mp2_time_s"] = time.time() - t1
            total_hartree += e_corr
        else:  # ccsd or ccsd(t)
            t1 = time.time()
            mycc = cc.CCSD(mf, frozen=frozen) if frozen else cc.CCSD(mf)
            mycc.conv_tol = float(params["conv_tol"])
            mycc.kernel()
            if not mycc.converged:
                logging.warning("PySCF CCSD did not converge")
            e_ccsd_corr = mycc.e_corr
            self.results["ccsd_correlation_eV"] = e_ccsd_corr * _HARTREE_TO_EV
            self.results["ccsd_time_s"] = time.time() - t1
            total_hartree += e_ccsd_corr
            if method == "ccsd(t)":
                t2 = time.time()
                e_t = mycc.ccsd_t()
                self.results["t_correction_eV"] = e_t * _HARTREE_TO_EV
                self.results["t_time_s"] = time.time() - t2
                total_hartree += e_t

        self.results["energy"] = total_hartree * _HARTREE_TO_EV
        self.results["free_energy"] = self.results["energy"]
        self.results["total_energy_eV"] = total_hartree * _HARTREE_TO_EV
