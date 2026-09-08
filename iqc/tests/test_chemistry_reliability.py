"""Regression tests for electronic state, stationary points, and recovery."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes, CalculationFailed

from iqc import asetools as at


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float((atoms.positions**2).sum()),
            "forces": -2 * atoms.positions.copy(),
        }


def test_charged_smiles_preserves_state_and_xyz_roundtrip():
    atoms = at.get_atoms_from_smiles("[NH4+]")
    assert atoms.info["charge"] == 1
    restored = at.get_atoms_from_xyz(at.atoms2xyz(atoms))
    assert restored.info["charge"] == 1
    assert at.get_multiplicity(restored, charge=1) == 1


@pytest.mark.parametrize("charge,mult", [(0.5, 1), (0, 1.5), (0, 2), (11, 1), (0, 13)])
def test_impossible_electronic_states_rejected(charge, mult):
    with pytest.raises(ValueError):
        at.get_multiplicity(molecule("H2O"), multiplicity=mult, charge=charge)


def test_changed_charge_invalidates_calculator_cache():
    class ORCA(HarmonicCalculator):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results["energy"] += self.parameters["charge"]

    atoms = molecule("H2O")
    calc = ORCA()
    at.apply_spin_charge(atoms, calc, charge=0)
    atoms.calc = calc
    neutral = atoms.get_potential_energy()
    at.apply_spin_charge(atoms, calc, charge=1)
    assert atoms.get_potential_energy() == pytest.approx(neutral + 1)


def test_exachem_reference_type_is_resolved_for_each_molecule():
    from iqc.exachem import ExaChemCalculator

    calc = ExaChemCalculator()
    atoms = molecule("H2O")
    at.apply_spin_charge(atoms, calc, charge=0)
    at.apply_spin_charge(atoms, calc, charge=1)
    payload = calc._build_input_json(atoms, calc.parameters, "scf")
    assert payload["SCF"]["scf_type"] == "unrestricted"


def test_unconverged_geometry_is_a_failed_result():
    atoms = molecule("H2O")
    _, result = at.run_optimization(atoms, calculator=HarmonicCalculator(), max_steps=0)
    assert result["opt_converged"] is False
    assert "converge" in result["error"]


def test_optimization_recovery_restores_last_valid_geometry(monkeypatch):
    atoms = Atoms("H", positions=[[1, 0, 0]], calculator=HarmonicCalculator())
    starts = []

    class Optimizer:
        def attach(self, observer, interval):
            self.observer = observer

        def run(self, **kwargs):
            starts.append(atoms.positions.copy())
            self.observer()
            if len(starts) == 1:
                atoms.positions[:] = np.nan
                raise np.linalg.LinAlgError("unstable Hessian")
            atoms.positions[:] = 0
            self.observer()
            return True

        def get_number_of_steps(self):
            return 1

    monkeypatch.setattr(at, "_make_optimizer", lambda *a, **kw: Optimizer())
    out = at._run_staged_optimization(atoms, 0.01, 10, None, recover=True)
    assert out[0] is True and out[3] is True
    assert np.allclose(starts[1], starts[0])
    assert np.isfinite(atoms.positions).all()


def test_large_negative_all_electron_energy_is_not_divergence(monkeypatch):
    class HeavyAtom(HarmonicCalculator):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results["energy"] -= 300000

    atoms = Atoms("I", positions=[[0, 0, 0]], calculator=HeavyAtom())
    assert at._run_staged_optimization(atoms, 0.01, 2, None)[0] is True


def test_imaginary_mode_before_zero_modes_is_not_discarded():
    frequencies = np.array([500j, 0, 0, 0, 0, 0, 0, 1000, 1500])
    modes = np.arange(81).reshape(9, 3, 3)
    vectors = at._imaginary_mode_vectors(frequencies, modes, 3, 50)
    assert len(vectors) == 1
    assert np.array_equal(vectors[0], modes[0])


def test_physical_failure_markers_remain_retryable():
    from iqc.main import _record_status
    from iqc.databasetools import _blob_is_success
    from iqc.status_query import _record_error

    for extra in ({"nonphysical": True}, {"opt_converged": False}):
        record = {"task": "opt", "error": "", **extra}
        assert _record_status(record) == "error"
        assert not _blob_is_success(json.dumps(record), "opt")
        assert _record_error(record)


def test_vibrations_without_optimization_applies_requested_state(tmp_path):
    class XTB(HarmonicCalculator):
        charges = []

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            self.charges.append(atoms.get_initial_charges().sum())
            super().calculate(atoms, properties, system_changes)

    calc = XTB()
    _, result = at.run_vibrations(
        molecule("H2O"),
        calculator=calc,
        optimize=False,
        charge=1,
        multiplicity=2,
        vib_dir=tmp_path,
    )
    assert not result["error"]
    assert calc.charges and set(calc.charges) == {1}


def test_composite_thermo_uses_separate_optimization_calculator(monkeypatch):
    atoms = molecule("H2O")
    opt, vib = HarmonicCalculator(), HarmonicCalculator()
    calls = []

    def optimize(a, calculator, **kwargs):
        calls.append(("opt", calculator))
        return a, {"opt_converged": True, "opt_sym_number": 2, "error": ""}

    def vibrations(a, calculator, optimize, **kwargs):
        calls.append(("vib", calculator, optimize))
        return a, {"error": "", "vib_energies": [0.1, 0.2, 0.3]}

    monkeypatch.setattr(at, "run_optimization", optimize)
    monkeypatch.setattr(at, "run_vibrations", vibrations)
    monkeypatch.setattr(
        at, "_add_thermo_results_from_vibrations", lambda a, r, **kw: (None, r)
    )
    at.run_thermo(atoms, optimization_calculator=opt, vibration_calculator=vib)
    assert calls == [("opt", opt), ("vib", vib, False)]


def test_saddle_hessian_is_detected_and_rejected_by_thermochemistry(tmp_path):
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
    atoms, result = at.run_vibrations(
        atoms, calculator=Saddle(), optimize=False, vib_dir=tmp_path
    )
    assert not result["error"]
    assert result["number_of_imaginary"] == 1
    assert result["vibrational_frequencies_cm^-1"][0] < 0
    thermo, result = at._add_thermo_results_from_vibrations(
        atoms, result, ignore_imag_modes=False
    )
    assert thermo is None and "Imaginary" in result["error"]


def test_imaginary_recovery_tries_opposite_direction_and_keeps_identity():
    atoms = Atoms("H", positions=[[0, 0, 0]])
    original = {
        "number_of_imaginary": 1,
        "initial_xyz": "original input",
        "unique_name": "original",
        "error": "",
    }
    trials = []

    def recompute(trial):
        trials.append(trial.positions.copy())
        if trial.positions[0, 0] > 0:
            return trial, {"error": "failed"}
        return trial, {
            "error": "",
            "number_of_imaginary": 0,
            "initial_xyz": "displaced input",
            "unique_name": "trial",
        }

    result = at._recover_imaginary(
        atoms, original, [np.array([[1, 0, 0]])], recompute, max_attempts=2
    )
    assert len(trials) == 2
    assert result["number_of_imaginary"] == 0
    assert result["initial_xyz"] == "original input"
    assert result["unique_name"] == "original"
    assert atoms.positions[0, 0] < 0


@pytest.mark.parametrize("value", ["charge=0.5", "multiplicity=1.5", "charge=NaN"])
def test_fractional_xyz_state_is_not_silently_truncated(value):
    with pytest.raises(ValueError):
        at.parse_multiplicity_charge_from_comment(value)


def test_partial_hessian_keeps_modes_but_cannot_supply_molecular_thermo():
    atoms = molecule("H2O")
    result = {"warnings": [], "error": "", "vib_energies": [0.1j, 0.2, 0.3]}
    at._record_vibrational_analysis(atoms, result, [100j, 200, 300], None, 100, 50)
    assert result["number_of_imaginary"] == 1
    assert len(result["thermo_vib_energies"]) == 3
    thermo, result = at._add_thermo_results_from_vibrations(atoms, result)
    assert thermo is None
    assert "complete molecular Hessian" in result["error"]


@pytest.mark.parametrize("dispatch", ["parsl_dispatch", "ensemble_launcher_dispatch"])
@pytest.mark.parametrize("charge,mult", [(0, 1), (1, 2), (0, 3)])
def test_serial_and_worker_failure_keys_match_requested_state(
    tmp_path, dispatch, charge, mult
):
    import importlib

    if dispatch == "parsl_dispatch":
        pytest.importorskip("parsl")
    from iqc.cli import get_args
    from iqc.main import _process_one_row, SKIPPED_EXISTING
    from iqc.databasetools import calculation_key_from_record

    driver = importlib.import_module(f"iqc.{dispatch}")
    xyz = tmp_path / "water.xyz"
    xyz.write_text(at.atoms2xyz(molecule("H2O")))
    args = get_args(["--xyz", str(xyz), "--skip-existing"])
    calc_params = {"charge": charge, "multiplicity": mult}
    params_str = json.dumps({"calculator_params": calc_params})
    kwargs = dict(
        args=args,
        params_str=params_str,
        task="single",
        calculator_name="test",
        xyz_files=[str(xyz)],
        input_mode="xyz",
        number_of_files=1,
    )
    run_kwargs = dict(
        **kwargs,
        calc_params=calc_params,
        opt_params={},
        vib_params={},
        ir_params={},
        thermo_params={},
        nmr_params={},
        calculator=HarmonicCalculator(),
        worker_id=0,
        n_workers=1,
        rank_output_dir_factory=lambda: str(tmp_path),
        direct_work_dir=str(tmp_path),
        completed_file_index={},
        db_path=None,
    )
    success = _process_one_row(0, **run_kwargs)
    assert not success.get("single_error") and not success.get("error")
    failure = driver._synthesize_failure_row(0, RuntimeError("worker lost"), **kwargs)
    key = calculation_key_from_record(success)
    assert calculation_key_from_record(failure) == key
    run_kwargs["completed_file_index"] = {key: "ok"}
    assert _process_one_row(0, **run_kwargs) is SKIPPED_EXISTING
    run_kwargs["calc_params"] = {"charge": 0, "multiplicity": 2}
    invalid = _process_one_row(0, **run_kwargs)
    assert "incompatible" in invalid["single_error"]
    if charge == 0 and mult == 1:
        assert success["initial_xyz"].splitlines()[1] == ""
    else:
        assert (
            success["initial_xyz"].splitlines()[1]
            == f"charge={charge} multiplicity={mult}"
        )
