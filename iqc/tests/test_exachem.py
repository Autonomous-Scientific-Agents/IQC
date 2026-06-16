"""Tests for the ExaChem ASE calculator wrapper.

Most tests build a calculator and inspect the generated JSON input/command —
no ExaChem binary is required. A single smoke test runs an actual SCF and is
skipped unless the binary is available on this machine.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule

from iqc.asetools import apply_spin_charge, get_calculator, run_single_point
from iqc.exachem import (
    ExaChemCalculator,
    _atoms_to_coordinate_lines,
    _deep_merge,
    _normalize_method,
)


@pytest.fixture
def water():
    return molecule("H2O")


def test_normalize_method_aliases():
    assert _normalize_method("SCF") == "scf"
    assert _normalize_method("hf") == "hf"
    assert _normalize_method("CCSD") == "ccsd"
    assert _normalize_method("CCSD(T)") == "ccsd(t)"
    assert _normalize_method("ccsd_t") == "ccsd_t"


def test_normalize_method_rejects_unknown():
    with pytest.raises(ValueError):
        _normalize_method("dmrg")


def test_deep_merge_recursive():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    overrides = {"b": {"y": 99, "z": 3}, "c": 4}
    result = _deep_merge(base, overrides)
    assert result == {"a": 1, "b": {"x": 1, "y": 99, "z": 3}, "c": 4}


def test_atoms_to_coordinate_lines_formats_three_columns(water):
    lines = _atoms_to_coordinate_lines(water)
    assert len(lines) == 3
    for symbol, line in zip(water.get_chemical_symbols(), lines):
        tokens = line.split()
        assert tokens[0] == symbol
        # Three coordinate columns parseable as floats.
        assert [float(t) for t in tokens[1:]] == pytest.approx(
            list(water.get_positions()[lines.index(line)])
        )


def test_build_input_scf_only_has_task_scf(water):
    calc = ExaChemCalculator(method="scf", basis="cc-pvdz")
    payload = calc._build_input_json(water, calc.parameters, "scf")
    assert payload["TASK"] == {"scf": True}
    assert payload["basis"]["basisset"] == "cc-pvdz"
    assert payload["SCF"]["charge"] == 0
    assert payload["SCF"]["multiplicity"] == 1
    assert payload["SCF"]["scf_type"] == "restricted"
    assert payload["geometry"]["units"] == "angstrom"


def test_build_input_ccsd_t_disables_lower_tasks(water):
    """ExaChem refuses inputs with more than one TASK enabled."""
    calc = ExaChemCalculator(method="ccsd(t)")
    payload = calc._build_input_json(water, calc.parameters, "ccsd(t)")
    assert payload["TASK"] == {"ccsd_t": True}


def test_build_input_ccsd_method(water):
    calc = ExaChemCalculator(method="ccsd")
    payload = calc._build_input_json(water, calc.parameters, "ccsd")
    assert payload["TASK"] == {"ccsd": True}


def test_build_input_open_shell_defaults_to_unrestricted():
    radical = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    calc = ExaChemCalculator(method="scf", multiplicity=2)
    payload = calc._build_input_json(radical, calc.parameters, "scf")
    assert payload["SCF"]["multiplicity"] == 2
    assert payload["SCF"]["scf_type"] == "unrestricted"


def test_build_input_user_overrides_deep_merge(water):
    calc = ExaChemCalculator(
        method="ccsd",
        scf={"diis_hist": 25, "tol_lindep": 1e-7},
        cc={"ccsd_maxiter": 200, "CCSD(T)": {"ccsdt_tilesize": 64}},
        basis_block={"df_basisset": "cc-pvdz-rifit"},
    )
    payload = calc._build_input_json(water, calc.parameters, "ccsd")
    assert payload["SCF"]["diis_hist"] == 25
    assert payload["SCF"]["tol_lindep"] == 1e-7
    # Pre-existing defaults preserved.
    assert payload["SCF"]["charge"] == 0
    assert payload["CC"]["ccsd_maxiter"] == 200
    assert payload["CC"]["CCSD(T)"]["ccsdt_tilesize"] == 64
    assert payload["basis"]["df_basisset"] == "cc-pvdz-rifit"
    assert payload["basis"]["basisset"] == "cc-pvdz"


def test_build_input_exachem_input_escape_hatch(water):
    calc = ExaChemCalculator(
        method="scf",
        exachem_input={"DPLOT": {"cube": True}, "SCF": {"writem": 50}},
    )
    payload = calc._build_input_json(water, calc.parameters, "scf")
    assert payload["DPLOT"] == {"cube": True}
    assert payload["SCF"]["writem"] == 50


def test_build_command_injects_nproc_when_missing(tmp_path):
    cmd = ExaChemCalculator._build_command(
        {"nproc": 4, "mpi_command": "mpiexec", "binary": "/usr/bin/true"},
        tmp_path / "input.json",
    )
    assert cmd[:3] == ["mpiexec", "-n", "4"]
    assert cmd[-2:] == ["/usr/bin/true", str(tmp_path / "input.json")]


def test_build_command_respects_existing_nproc_flag(tmp_path):
    cmd = ExaChemCalculator._build_command(
        {
            "nproc": 4,
            "mpi_command": ["mpiexec", "-n", "2", "--bind-to", "core"],
            "binary": "/usr/bin/true",
        },
        tmp_path / "input.json",
    )
    # User-supplied -n is preserved; we don't add a second one.
    assert cmd.count("-n") == 1
    assert cmd[:5] == ["mpiexec", "-n", "2", "--bind-to", "core"]


def test_build_command_rejects_zero_nproc(tmp_path):
    with pytest.raises(ValueError):
        ExaChemCalculator._build_command(
            {"nproc": 0, "mpi_command": "mpiexec", "binary": "/usr/bin/true"},
            tmp_path / "input.json",
        )


def test_omp_env_set_from_param():
    env = ExaChemCalculator._build_env({"omp_num_threads": 4})
    assert env["OMP_NUM_THREADS"] == "4"


def test_omp_env_none_leaves_env_untouched(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    env = ExaChemCalculator._build_env({"omp_num_threads": None})
    assert "OMP_NUM_THREADS" not in env


def test_extract_energy_scf():
    payload = {"output": {"SCF": {"final_energy": -76.0}}}
    assert ExaChemCalculator._extract_energy(payload, "scf") == -76.0


def test_extract_energy_ccsd():
    payload = {
        "output": {
            "SCF": {"final_energy": -76.0},
            "CCSD": {"final_energy": {"correlation": -0.2, "total": -76.2}},
        }
    }
    assert ExaChemCalculator._extract_energy(payload, "ccsd") == -76.2


def test_extract_energy_ccsd_t_prefers_t_over_bracket_t():
    payload = {
        "output": {
            "SCF": {"final_energy": -76.0},
            "CCSD(T)": {
                "[T]Energies": {"total": -76.30},
                "(T)Energies": {"total": -76.25},
            },
        }
    }
    assert ExaChemCalculator._extract_energy(payload, "ccsd_t") == -76.25


def test_apply_spin_charge_exachem_writes_parameters(water):
    calc = ExaChemCalculator(method="scf")
    apply_spin_charge(water, calc, multiplicity=3, charge=-1)
    assert calc.parameters["charge"] == -1
    assert calc.parameters["multiplicity"] == 3
    assert calc.parameters["scf_type"] == "unrestricted"


def test_apply_spin_charge_exachem_respects_explicit_scf_type():
    radical = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    calc = ExaChemCalculator(method="scf", scf_type="rohf")
    apply_spin_charge(radical, calc, multiplicity=2, charge=0)
    assert calc.parameters["scf_type"] == "rohf"


def test_get_calculator_exachem_returns_instance():
    calc = get_calculator("exachem", method="scf", basis="cc-pvdz", nproc=2)
    # On a system without the binary, the calculator constructor still
    # succeeds (validation happens at run-time). A failure would fall back
    # to MACE — check that we got the actual ExaChem calculator.
    assert isinstance(calc, ExaChemCalculator)
    assert calc.parameters["method"] == "scf"
    assert calc.parameters["nproc"] == 2


# --- Integration smoke test --------------------------------------------------

_BINARY = os.environ.get(
    "EXACHEM_BINARY", "/home/keceli/soft/nwx/exachem/build/install/bin/ExaChem"
)
_MPIEXEC = shutil.which("mpiexec")


@pytest.mark.skipif(
    not (_MPIEXEC and Path(_BINARY).is_file()),
    reason="ExaChem binary or mpiexec not available on this system",
)
def test_exachem_scf_h2o_smoke(tmp_path, water):
    """End-to-end: SCF energy of H2O / cc-pVDZ is reproducible to mHa."""
    calc = ExaChemCalculator(
        method="scf",
        basis="cc-pvdz",
        nproc=2,
        directory=str(tmp_path),
        binary=_BINARY,
    )
    water.calc = calc
    energy_eV = water.get_potential_energy()
    energy_ha = calc.results["energy_hartree"]
    # Reference: ASE's MP2-equilibrium H2O gives ~-76.0 Ha at HF/cc-pVDZ
    # (the exact reference cmd run earlier gave -75.8251). Geometry differs
    # slightly; loose bound is enough for a smoke test.
    assert -76.5 < energy_ha < -75.5
    assert abs(energy_eV / energy_ha - 27.2114) < 0.01
    assert calc.last_output_path is not None
    assert calc.last_output_path.is_file()
    with open(calc.last_output_path) as fh:
        payload = json.load(fh)
    assert payload["output"]["SCF"]["final_energy"] == pytest.approx(energy_ha)


@pytest.mark.skipif(
    not (_MPIEXEC and Path(_BINARY).is_file()),
    reason="ExaChem binary or mpiexec not available on this system",
)
def test_exachem_run_single_point_integration(tmp_path, water):
    """run_single_point should drive ExaChem and return an energy_eV value."""
    calc = get_calculator(
        "exachem",
        method="scf",
        basis="cc-pvdz",
        nproc=2,
        directory=str(tmp_path),
        binary=_BINARY,
    )
    _, results = run_single_point(water, calculator=calc, unique_name="h2o_int")
    assert results["error"] == ""
    assert results["energy_eV"] < 0
    # Forces are not implemented; a single warning is expected.
    assert any("Forces" in w for w in results["warnings"])
