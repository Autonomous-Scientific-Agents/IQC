"""Tests for rigid-body (translation/rotation) projection of Hessians."""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations, VibrationsData

from iqc import asetools as at
from iqc.hessiantools import (
    eigenvalues_to_frequencies_cm,
    project_hessian,
    project_vibrations_data,
    rigid_body_basis,
)

# --------------------------------------------------------------------------
# Synthetic, exactly invariant Hessians
# --------------------------------------------------------------------------


def _spring_hessian(atoms, k):
    """Cartesian Hessian of all-pairs harmonic springs at their rest lengths.

    Exactly translationally and rotationally invariant, so the rigid-body
    eigenvalues are zero to machine precision. ``k`` is a scalar or a dict
    ``{(i, j): k_ij}``.
    """
    n = len(atoms)
    x = atoms.get_positions()
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            kij = k[(i, j)] if isinstance(k, dict) else k
            u = x[j] - x[i]
            u /= np.linalg.norm(u)
            block = kij * np.outer(u, u)
            for a, b, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
                H[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] += sign * block
    return H


def _sqrt_masses(atoms):
    return np.sqrt(np.repeat(atoms.get_masses(), 3))


def _add_rigid_body_contamination(H, atoms, freqs_cm):
    """Add stiffness purely inside the rigid-body subspace (model of a
    non-invariant Hessian from grid noise or a residual gradient)."""
    D, n_rigid = rigid_body_basis(atoms)
    freqs_cm = np.broadcast_to(np.asarray(freqs_cm, float), n_rigid)
    ev = (freqs_cm / eigenvalues_to_frequencies_cm(1.0)) ** 2
    sm = _sqrt_masses(atoms)
    return H + (D * ev) @ D.T * sm[:, None] * sm[None, :]


def _freqs(H, atoms):
    inv = 1.0 / _sqrt_masses(atoms)
    return eigenvalues_to_frequencies_cm(
        np.linalg.eigvalsh(H * inv[:, None] * inv[None, :])
    )


def test_frequency_conversion_matches_ase():
    atoms = molecule("H2O")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(9, 9))
    H = A @ A.T
    inv = 1.0 / _sqrt_masses(atoms)
    mine = eigenvalues_to_frequencies_cm(
        np.linalg.eigvalsh(H * inv[:, None] * inv[None, :])
    )
    ase_freqs = VibrationsData.from_2d(atoms, H).get_frequencies()
    np.testing.assert_allclose(np.sort(mine), np.sort(ase_freqs.real), rtol=1e-8)


def test_rigid_body_basis_is_orthonormal_and_rank_aware():
    for name, expected in (("H2O", 6), ("CO2", 5), ("CH4", 6)):
        atoms = molecule(name)
        D, n_rigid = rigid_body_basis(atoms)
        assert n_rigid == expected
        assert D.shape == (3 * len(atoms), n_rigid)
        np.testing.assert_allclose(D.T @ D, np.eye(n_rigid), atol=1e-12)
    D, n_rigid = rigid_body_basis(Atoms("Ar", positions=[[0, 0, 0]]))
    assert n_rigid == 3 and D.shape == (3, 3)


def test_requesting_more_rotations_than_present_falls_back(caplog):
    co2 = molecule("CO2")
    with caplog.at_level(logging.WARNING):
        _, n_rigid = rigid_body_basis(co2, n_rot=3)
    assert n_rigid == 5
    assert "only 2 are numerically present" in caplog.text
    with pytest.raises(ValueError):
        rigid_body_basis(co2, n_rot=4)


def test_projection_removes_contamination_and_recovers_soft_mode():
    """A genuine 30 cm^-1 mode must survive even when the rigid-body block
    is contaminated at 80 cm^-1 - the case magnitude ordering gets wrong."""
    atoms = molecule("H2O")
    # Weak O-H springs plus a very weak H-H spring give one soft internal mode.
    k = {(0, 1): 30.0, (0, 2): 30.0, (1, 2): 0.004}
    H_clean = _spring_hessian(atoms, k)
    clean = np.sort(_freqs(H_clean, atoms))
    soft = clean[6]
    assert 20 < soft < 40, soft
    np.testing.assert_allclose(clean[:6], 0.0, atol=1e-3)

    H_dirty = _add_rigid_body_contamination(H_clean, atoms, 80.0)
    dirty = np.sort(np.abs(_freqs(H_dirty, atoms)))
    # Magnitude ordering discards the soft mode and keeps a rotation.
    _, selected = at._vibrational_mode_indices(_freqs(H_dirty, atoms), 3)
    assert not np.any(
        np.isclose(np.abs(_freqs(H_dirty, atoms))[selected], soft, atol=1.0)
    )
    assert np.isclose(dirty[3], 80.0, atol=1e-3)

    H_proj, report = project_hessian(H_dirty, atoms)
    projected = np.sort(report["frequencies_projected_cm"])
    np.testing.assert_allclose(projected[:6], 0.0, atol=1e-3)
    np.testing.assert_allclose(projected[6:], clean[6:], rtol=1e-9)
    np.testing.assert_allclose(report["rigid_body_frequencies_cm"], 80.0, atol=1e-3)
    assert report["n_rigid"] == 6 and report["geometry"] == "nonlinear"
    # Returned matrix is a Cartesian Hessian usable by ASE directly.
    ase_proj = VibrationsData.from_2d(atoms, H_proj).get_frequencies().real
    np.testing.assert_allclose(np.sort(ase_proj)[6:], clean[6:], rtol=1e-9)


def test_projection_is_idempotent_on_clean_hessian():
    atoms = molecule("CH4")
    H = _spring_hessian(atoms, 25.0)
    H_proj, report = project_hessian(H, atoms)
    np.testing.assert_allclose(H_proj, H, atol=1e-9)
    assert report["rigid_body_coupling_norm"] < 1e-10
    np.testing.assert_allclose(report["rigid_body_frequencies_cm"], 0.0, atol=1e-3)


def test_linear_molecule_keeps_five_rigid_modes_and_bends():
    co2 = molecule("CO2")
    H = _add_rigid_body_contamination(_spring_hessian(co2, 40.0), co2, 50.0)
    _, report = project_hessian(H, co2)
    assert report["n_rigid"] == 5 and report["geometry"] == "linear"
    projected = np.sort(np.abs(report["frequencies_projected_cm"]))
    # Five zeros, then two degenerate (zero-stiffness) bends and two stretches.
    np.testing.assert_allclose(projected[:5], 0.0, atol=1e-3)
    assert len(report["rigid_body_frequencies_cm"]) == 5


def test_monatomic_projection():
    ar = Atoms("Ar", positions=[[0, 0, 0]])
    H_proj, report = project_hessian(np.eye(3) * 5.0, ar)
    assert report["n_rigid"] == 3 and report["geometry"] == "monatomic"
    np.testing.assert_allclose(H_proj, 0.0, atol=1e-12)


def test_project_hessian_accepts_4d_and_rejects_bad_input():
    atoms = molecule("H2O")
    H = _spring_hessian(atoms, 10.0)
    H4 = H.reshape(3, 3, 3, 3)
    a, _ = project_hessian(H, atoms)
    b, _ = project_hessian(H4, atoms)
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError, match="does not match"):
        project_hessian(H[:6, :6], atoms)
    bad = H.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        project_hessian(bad, atoms)


def test_project_vibrations_data_accepts_ase_objects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    atoms = molecule("H2O")
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-4)
    vib = Vibrations(atoms, name="vib_h2o")
    vib.run()
    from_vib, rep1 = project_vibrations_data(vib)
    from_data, rep2 = project_vibrations_data(vib.get_vibrations())
    assert isinstance(from_vib, VibrationsData)
    np.testing.assert_allclose(from_vib.get_hessian_2d(), from_data.get_hessian_2d())
    assert rep1["n_rigid"] == rep2["n_rigid"] == 6
    projected = np.sort(np.abs(from_data.get_frequencies()))
    np.testing.assert_allclose(projected[:6], 0.0, atol=1e-4)
    # Internal modes are unchanged to within the (small) contamination.
    raw = np.sort(np.abs(vib.get_frequencies()))
    np.testing.assert_allclose(projected[6:], raw[6:], rtol=1e-3)

    partial = Vibrations(atoms, name="vib_partial", indices=[1, 2])
    partial.run()
    with pytest.raises(ValueError, match="complete Hessian"):
        project_vibrations_data(partial.get_vibrations())
    with pytest.raises(TypeError):
        project_vibrations_data(np.eye(9))


# --------------------------------------------------------------------------
# Pipeline integration
# --------------------------------------------------------------------------


class RigidBodyContaminated(Calculator):
    """EMT plus a stiffness acting only on rigid-body displacements from a
    reference geometry. Forces vanish at the reference, so it is a stationary
    point, but the Hessian's translational/rotational block is lifted to
    ``freq_cm`` - the signature of a non-invariant force field, grid noise,
    or a mis-specified finite-difference step."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, reference, freq_cm, **kwargs):
        super().__init__(**kwargs)
        self._emt = EMT()
        self._ref = reference.get_positions().copy()
        D, _ = rigid_body_basis(reference)
        ev = (freq_cm / eigenvalues_to_frequencies_cm(1.0)) ** 2
        sm = _sqrt_masses(reference)
        self._K = (ev * D) @ D.T * sm[:, None] * sm[None, :]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._emt.calculate(atoms, ["energy", "forces"], all_changes)
        delta = (atoms.get_positions() - self._ref).ravel()
        grad = self._K @ delta
        self.results = {
            "energy": float(self._emt.results["energy"] + 0.5 * delta @ grad),
            "forces": self._emt.results["forces"] - grad.reshape(-1, 3),
        }


@pytest.fixture
def h2o_emt_minimum():
    atoms = molecule("H2O")
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-5)
    atoms.calc = None
    return atoms


def _run_vib(atoms, calc, tmp_path, **kwargs):
    with patch("builtins.print"):  # ASE's summary() writes to stdout
        return at.run_vibrations(
            atoms.copy(),
            calculator=calc,
            optimize=False,
            vib_dir=str(tmp_path),
            **kwargs,
        )


def test_run_vibrations_projects_by_default_and_reports_contamination(
    h2o_emt_minimum, tmp_path
):
    _, clean = _run_vib(h2o_emt_minimum, EMT(), tmp_path, project_trans_rot=False)
    assert not clean["error"]
    clean_vib = np.sort(clean["vibrational_frequencies_cm^-1"])

    calc = RigidBodyContaminated(h2o_emt_minimum, freq_cm=300.0)
    _, res = _run_vib(h2o_emt_minimum, calc, tmp_path)
    assert not res["error"], res["error"]
    assert res["trans_rot_projected"] is True
    assert len(res["trans_rot_frequencies_cm^-1"]) == 6
    np.testing.assert_allclose(
        np.abs(res["trans_rot_frequencies_cm^-1"]), 300.0, rtol=0.05
    )
    assert any("too high" in w for w in res["warnings"])
    assert res["vibration_complete"] is True
    assert len(res["vibrational_frequencies_cm^-1"]) == 3
    assert len(res["thermo_vib_energies"]) == 3
    # Contamination confined to the rigid-body block: internal modes recovered.
    np.testing.assert_allclose(
        np.sort(res["vibrational_frequencies_cm^-1"]), clean_vib, rtol=2e-3, atol=0.5
    )
    # The 3N raw list carries the projected (zero) externals.
    raw = np.sort(np.abs(np.asarray(res["frequencies_cm^-1"], dtype=complex)))
    assert raw[5] < 1.0

    _, unprojected = _run_vib(h2o_emt_minimum, calc, tmp_path, project_trans_rot=False)
    assert unprojected["trans_rot_projected"] is False
    assert "trans_rot_frequencies_cm^-1" not in unprojected
    # Magnitude ordering keeps ~300 cm^-1 rotations and drops the real bend.
    wrong = np.sort(unprojected["vibrational_frequencies_cm^-1"])
    assert wrong[0] > clean_vib[0] + 50
    assert any("too high" in w for w in unprojected["warnings"])


def test_run_vibrations_partial_hessian_skips_projection(h2o_emt_minimum, tmp_path):
    _, res = _run_vib(h2o_emt_minimum, EMT(), tmp_path, indices=[1, 2])
    assert not res["error"], res["error"]
    assert res["vibration_complete"] is False
    assert res["trans_rot_projected"] is False
    assert any("projection skipped" in w for w in res["warnings"])
    assert len(res["vibrational_frequencies_cm^-1"]) == 6


def test_run_vibrations_survives_projection_failure(h2o_emt_minimum, tmp_path):
    with patch(
        "iqc.asetools.project_vibrations_data", side_effect=RuntimeError("boom")
    ):
        _, res = _run_vib(h2o_emt_minimum, EMT(), tmp_path)
    assert not res["error"]
    assert res["trans_rot_projected"] is False
    assert any("projection failed: boom" in w for w in res["warnings"])
    assert len(res["vibrational_frequencies_cm^-1"]) == 3


def test_run_thermo_forwards_projection_flag(h2o_emt_minimum, tmp_path):
    with patch("builtins.print"):
        _, on = at.run_thermo(
            h2o_emt_minimum.copy(),
            calculator=EMT(),
            optimize=False,
            vib_dir=str(tmp_path),
        )
        _, off = at.run_thermo(
            h2o_emt_minimum.copy(),
            calculator=EMT(),
            optimize=False,
            vib_dir=str(tmp_path),
            project_trans_rot=False,
        )
    assert not on["error"] and not off["error"]
    assert on["trans_rot_projected"] is True
    assert off["trans_rot_projected"] is False
    assert np.isfinite(on["G_eV"]) and np.isfinite(off["G_eV"])
    # Clean EMT minimum: both selections agree.
    assert abs(on["G_eV"] - off["G_eV"]) < 1e-3


def test_run_ir_projects_modes(h2o_emt_minimum, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Dipole:
        name = "dipole"

        def get_dipole_moment(self, atoms):
            return np.array([0.0, 0.0, 0.1])

    with patch("builtins.print"):
        _, res = at.run_ir(
            h2o_emt_minimum.copy(),
            vibration_calculator=EMT(),
            dipole_calculator=Dipole(),
            optimize=False,
            unique_name="h2o_ir",
            vib_dir=str(tmp_path / "ir"),
        )
    assert not res["error"], res["error"]
    assert res["trans_rot_projected"] is True
    assert len(res["trans_rot_frequencies_cm^-1"]) == 6
    assert len(res["vibrational_frequencies_cm^-1"]) == 3


def test_jmol_export_matches_ase_layout_and_uses_projected_modes(tmp_path, monkeypatch):
    import io

    monkeypatch.chdir(tmp_path)
    atoms = molecule("H2O")
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-4)
    vib = Vibrations(atoms, name="vib_jmol")
    vib.run()
    legacy = io.StringIO()
    vib._write_jmol(legacy)
    assert at._jmol_modes_text(vib.get_vibrations()) == legacy.getvalue()

    projected, _ = project_vibrations_data(vib)
    text = at._jmol_modes_text(projected)
    headers = [line for line in text.splitlines() if line.startswith("Mode #")]
    assert len(headers) == 9
    # Six rigid-body modes are exactly zero after projection.
    zeros = [h for h in headers if "f = 0.0" in h or "f = -0.0" in h]
    assert len(zeros) == 6


def test_imaginary_recovery_forwards_projection_flag(tmp_path):
    class Saddle(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            vector = atoms.positions[1] - atoms.positions[0]
            distance = np.linalg.norm(vector)
            force = (distance - 0.74) * vector / distance
            self.results = {
                "energy": -0.5 * (distance - 0.74) ** 2,
                "forces": np.array([-force, force]),
            }

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    real_run_vibrations = at.run_vibrations
    with patch.object(at, "run_vibrations") as recursive, patch("builtins.print"):
        recursive.return_value = (atoms, {"error": "trial failed"})
        _, res = real_run_vibrations(
            atoms,
            calculator=Saddle(),
            optimize=False,
            vib_dir=str(tmp_path),
            imag_recovery=True,
            project_trans_rot=False,
        )
    assert res["number_of_imaginary"] == 1
    assert recursive.called
    assert recursive.call_args.kwargs["project_trans_rot"] is False
