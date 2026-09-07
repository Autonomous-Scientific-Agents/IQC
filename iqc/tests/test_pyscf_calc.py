"""Exercise PySCF convergence contracts without requiring PySCF in CI."""

from types import SimpleNamespace
import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import CalculationFailed
from iqc.pyscf_calc import PySCFCalculator


class MeanField:
    def __init__(self, converged=False, recovered=None):
        self.converged = converged
        self.recovered = recovered
        self.newton_calls = 0
        self.conv_tol = 1e-9
        self.max_cycle = 200

    def kernel(self, **kwargs):
        return -1.0 if self.converged else -0.5

    def newton(self):
        self.newton_calls += 1
        return self.recovered or self

    def make_rdm1(self):
        return np.eye(2)

    def nuc_grad_method(self):
        return SimpleNamespace(kernel=lambda: np.zeros((2, 3)))

    def dip_moment(self, **kwargs):
        return [0, 0, 0]


def attach_reference(monkeypatch, calc, mf):
    monkeypatch.setattr(calc, "_build_mol", lambda: (object(), 0))
    monkeypatch.setattr(calc, "_make_scf", lambda *a: mf)


def test_scf_failure_never_returns_energy(monkeypatch):
    calc = PySCFCalculator(method="hf", scf_recovery=False)
    mf = MeanField()
    attach_reference(monkeypatch, calc, mf)
    with pytest.raises(CalculationFailed, match="converge"):
        calc.get_potential_energy(Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]))
    assert "energy" not in calc.results
    assert mf.newton_calls == 0


def test_scf_newton_recovery_is_bounded_and_recorded(monkeypatch):
    calc = PySCFCalculator(method="hf")
    mf = MeanField(recovered=MeanField(converged=True))
    attach_reference(monkeypatch, calc, mf)
    energy = calc.get_potential_energy(Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]))
    assert energy == pytest.approx(-units.Hartree)
    assert mf.newton_calls == 1
    assert calc.results["scf_recovery_used"] is True
    assert calc.results["scf_converged"] is True


def test_exhausted_scf_recovery_raises_and_clears_results(monkeypatch):
    calc = PySCFCalculator(method="hf")
    mf = MeanField()
    attach_reference(monkeypatch, calc, mf)
    with pytest.raises(CalculationFailed, match="converge"):
        calc.calculate(Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]))
    assert mf.newton_calls == 1
    assert calc.results == {}


def test_changed_method_invalidates_results_and_updates_capabilities():
    calc = PySCFCalculator(method="hf", basis="sto-3g")
    calc.results = {"energy": -10.0}
    calc.set(method="ccsd(t)")
    assert calc.results == {}
    assert "forces" not in calc.implemented_properties
    assert "ccsd(t)" in calc.model_name


def test_nonconverged_ccsd_does_not_run_triples(monkeypatch):
    import sys

    cc_solver = SimpleNamespace(
        converged=False,
        e_corr=-0.1,
        kernel=lambda: None,
        ccsd_t=lambda: pytest.fail("Triples must not run"),
    )
    mf = MeanField(converged=True)
    fake_pyscf = SimpleNamespace(
        cc=SimpleNamespace(CCSD=lambda *a, **kw: cc_solver),
        mp=SimpleNamespace(),
        scf=SimpleNamespace(
            uhf=SimpleNamespace(UHF=MeanField), rhf=SimpleNamespace(RHF=MeanField)
        ),
    )
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    calc = PySCFCalculator(method="ccsd(t)")
    attach_reference(monkeypatch, calc, mf)
    with pytest.raises(CalculationFailed, match="CCSD did not converge"):
        calc.calculate(Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]))
    assert calc.results == {}


def test_real_pyscf_neutral_ion_and_newton_recovery():
    pyscf = pytest.importorskip("pyscf")
    pyscf.lib.num_threads(1)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    calc = PySCFCalculator(method="hf", basis="sto-3g", max_cycle=1)
    neutral = calc.get_potential_energy(atoms)
    assert neutral / units.Hartree == pytest.approx(-1.1167593074, abs=1e-8)
    assert calc.results["scf_recovery_used"] is True
    calc.set(charge=1, multiplicity=2, max_cycle=200)
    ion = calc.get_potential_energy(atoms)
    assert ion / units.Hartree == pytest.approx(-0.5382054476, abs=1e-8)
    calc.set(method="uhf", charge=0, multiplicity=1)
    calc.get_potential_energy(atoms)
    mol, spin = calc._build_mol()
    assert isinstance(calc._make_scf(mol, spin), pyscf.scf.uhf.UHF)
