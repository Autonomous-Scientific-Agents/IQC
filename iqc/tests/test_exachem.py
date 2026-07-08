"""Tests for the ExaChem ASE calculator wrapper.

Most tests build a calculator and inspect the generated JSON input/command —
no ExaChem binary is required. A single smoke test runs an actual SCF and is
skipped unless the binary is available on this machine.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
from pathlib import Path
from unittest import mock

import pytest
from ase import Atoms, units
from ase.build import molecule

from iqc.asetools import _store_calculator_observables, apply_spin_charge, get_calculator, run_single_point
from iqc.exachem import (
    ExaChemCalculator,
    _atoms_to_coordinate_lines,
    _classify_file_kind,
    _deep_merge,
    _normalize_method,
    _stage_restart_artifacts,
    resolve_restart_artifact,
)


_HARTREE_TO_EV = units.Hartree


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


# --- Energy-component extraction --------------------------------------------

# Representative ExaChem CCSD(T) payload, mirroring the shape of the
# reference outputs shipped with the ExaChem repository (e.g.
# uracil.cc-pvdz.ccsd_t.json). Energies are in Hartree, timings in seconds.
#
# Real-fixture cross-check (verified locally against
# /lus/flare/projects/IQC/keceli/exachem/ci/reference_output/uracil.cc-pvdz.ccsd_t.json):
# the field paths used by ``_extract_components`` match. We embed the JSON
# inline here so the test suite has no external file dependency.
_CCSD_T_PAYLOAD = {
    "output": {
        "SCF": {
            "final_energy": -76.0,
            "performance": {"total_time": 1.5},
        },
        "CCSD": {
            "n_iterations": 10,
            "final_energy": {"correlation": -0.20, "total": -76.20},
            "performance": {"total_time": 55.0},
        },
        "CCSD(T)": {
            "[T]Energies": {"correction": -0.05, "correlation": -0.25, "total": -76.25},
            "(T)Energies": {"correction": -0.046, "correlation": -0.246, "total": -76.246},
            "performance": {"total_time": 12.5},
        },
    },
    "molecule": {"name": "uracil", "basis": {"basisset": "cc-pvdz"}},
    "input": {
        "basis": {"basisset": "cc-pvdz"},
        "SCF": {"scf_type": "restricted", "charge": 0, "multiplicity": 1},
        "CC": {"threshold": 1e-8, "freeze": {"atomic": True, "core": 0, "virtual": 0}},
        "TASK": {"ccsd_t": True},
    },
}


_SCF_ONLY_PAYLOAD = {
    "output": {
        "SCF": {
            "final_energy": -76.0,
            "performance": {"total_time": 1.0},
        }
    },
    "molecule": {"basis": {"basisset": "def2-tzvp"}},
    "input": {
        "basis": {"basisset": "def2-tzvp"},
        "SCF": {"scf_type": "unrestricted", "multiplicity": 2},
        "TASK": {"scf": True},
    },
}


_CCSD_FROZEN_PAYLOAD = {
    "output": {
        "SCF": {
            "final_energy": -229.29,
            "performance": {"total_time": 0.8},
        },
        "CCSD": {
            "final_energy": {"correlation": -0.32, "total": -229.61},
            "performance": {"total_time": 2.4},
        },
    },
    "molecule": {"basis": {"basisset": "sto-3g"}},
    "input": {
        "basis": {"basisset": "sto-3g"},
        "SCF": {"scf_type": "restricted"},
        "CC": {"freeze": {"atomic": True, "core": 0, "virtual": 0}},
        "TASK": {"ccsd": True},
    },
}


def _approx_ev(hartree):
    return pytest.approx(hartree * _HARTREE_TO_EV)


def test_extract_components_ccsd_t_full():
    """All energy components, timings, and method metadata extract for a (T) run."""
    comp = ExaChemCalculator._extract_components(_CCSD_T_PAYLOAD, "ccsd_t")

    # Energies (eV)
    assert comp["scf_energy_eV"] == _approx_ev(-76.0)
    assert comp["ccsd_correlation_eV"] == _approx_ev(-0.20)
    # The (T) correction is the asymmetric one, not [T].
    assert comp["t_correction_eV"] == _approx_ev(-0.046)
    # Total energy for a CCSD(T) run is the (T) total.
    assert comp["total_energy_eV"] == _approx_ev(-76.246)

    # MP2 is absent from this payload.
    assert comp["mp2_correlation_eV"] is None

    # Timings (seconds)
    assert comp["scf_time_s"] == pytest.approx(1.5)
    assert comp["ccsd_time_s"] == pytest.approx(55.0)
    assert comp["t_time_s"] == pytest.approx(12.5)

    # Method metadata
    assert comp["basis"] == "cc-pvdz"
    assert comp["scf_type"] == "restricted"
    assert comp["method"] == "ccsd_t"
    assert comp["frozen_core"] is True


def test_extract_components_ccsd_t_alias_method_normalizes():
    """A user-passed alias like ``ccsd(t)`` should still label method as ``ccsd_t``."""
    comp = ExaChemCalculator._extract_components(_CCSD_T_PAYLOAD, "ccsd(t)")
    assert comp["method"] == "ccsd_t"
    # Total still resolves via the (T)Energies block.
    assert comp["total_energy_eV"] == _approx_ev(-76.246)


def test_extract_components_scf_only_leaves_higher_methods_none():
    comp = ExaChemCalculator._extract_components(_SCF_ONLY_PAYLOAD, "scf")
    assert comp["scf_energy_eV"] == _approx_ev(-76.0)
    assert comp["total_energy_eV"] == _approx_ev(-76.0)
    assert comp["mp2_correlation_eV"] is None
    assert comp["ccsd_correlation_eV"] is None
    assert comp["t_correction_eV"] is None
    assert comp["ccsd_time_s"] is None
    assert comp["t_time_s"] is None
    assert comp["scf_time_s"] == pytest.approx(1.0)
    assert comp["basis"] == "def2-tzvp"
    assert comp["scf_type"] == "unrestricted"
    assert comp["method"] == "scf"
    assert comp["frozen_core"] is False


def test_extract_components_hf_alias_becomes_scf():
    comp = ExaChemCalculator._extract_components(_SCF_ONLY_PAYLOAD, "hf")
    assert comp["method"] == "scf"


def test_extract_components_ccsd_total_is_ccsd_total_not_scf():
    comp = ExaChemCalculator._extract_components(_CCSD_FROZEN_PAYLOAD, "ccsd")
    assert comp["scf_energy_eV"] == _approx_ev(-229.29)
    assert comp["ccsd_correlation_eV"] == _approx_ev(-0.32)
    assert comp["total_energy_eV"] == _approx_ev(-229.61)
    assert comp["t_correction_eV"] is None
    assert comp["frozen_core"] is True
    assert comp["basis"] == "sto-3g"
    assert comp["method"] == "ccsd"


def test_extract_components_mp2_scalar_final_energy():
    # TODO: replace with a real ExaChem MP2 output fixture once one is
    # captured locally — the ExaChem reference set on disk did not include
    # any *.mp2.json files. The shape below mirrors the path
    # ``_extract_energy`` already reads (``output.MP2.final_energy``).
    payload = {
        "output": {
            "SCF": {"final_energy": -76.0},
            "MP2": {"final_energy": -76.2},
        },
        "molecule": {"basis": {"basisset": "cc-pvdz"}},
        "input": {
            "basis": {"basisset": "cc-pvdz"},
            "SCF": {"scf_type": "restricted"},
            "TASK": {"mp2": True},
        },
    }
    comp = ExaChemCalculator._extract_components(payload, "mp2")
    assert comp["scf_energy_eV"] == _approx_ev(-76.0)
    assert comp["total_energy_eV"] == _approx_ev(-76.2)
    # Scalar final_energy doesn't carry a separate correlation breakdown.
    assert comp["mp2_correlation_eV"] is None
    assert comp["method"] == "mp2"


def test_extract_components_mp2_dict_final_energy_with_correlation():
    payload = {
        "output": {
            "SCF": {"final_energy": -76.0},
            "MP2": {
                "final_energy": {"correlation": -0.2, "total": -76.2},
            },
        },
        "molecule": {"basis": {"basisset": "cc-pvdz"}},
        "input": {"basis": {"basisset": "cc-pvdz"}, "TASK": {"mp2": True}},
    }
    comp = ExaChemCalculator._extract_components(payload, "mp2")
    assert comp["mp2_correlation_eV"] == _approx_ev(-0.2)
    assert comp["total_energy_eV"] == _approx_ev(-76.2)


def test_extract_components_missing_freeze_returns_false():
    payload = {
        "output": {"SCF": {"final_energy": -1.0}},
        "input": {
            "basis": {"basisset": "cc-pvdz"},
            "SCF": {"scf_type": "restricted"},
            "TASK": {"scf": True},
            # No CC block at all.
        },
    }
    comp = ExaChemCalculator._extract_components(payload, "scf")
    assert comp["frozen_core"] is False


def test_extract_components_no_basis_in_input_falls_back_to_molecule():
    payload = {
        "output": {"SCF": {"final_energy": -1.0}},
        "molecule": {"basis": {"basisset": "cc-pvtz"}},
        "input": {"TASK": {"scf": True}, "SCF": {"scf_type": "restricted"}},
    }
    comp = ExaChemCalculator._extract_components(payload, "scf")
    assert comp["basis"] == "cc-pvtz"


def test_extract_components_t_prefers_round_t_over_square_t():
    """ExaChem reports both [T] and (T); the canonical CCSD(T) total uses (T)."""
    comp = ExaChemCalculator._extract_components(_CCSD_T_PAYLOAD, "ccsd_t")
    # The [T] total in the fixture is -76.25; the (T) total is -76.246.
    assert comp["total_energy_eV"] == _approx_ev(-76.246)
    assert comp["t_correction_eV"] == _approx_ev(-0.046)


# --- F10: restart from prior amplitudes -------------------------------------


def _write_fake_artifact_files(root: Path):
    """Create a representative set of files an ExaChem run would emit."""

    (root / "h2o.cc-pvdz_files" / "restricted").mkdir(parents=True)
    base = root / "h2o.cc-pvdz_files" / "restricted"
    # Amplitudes
    (base / "h2o.cc-pvdz.ccsd.t1_amp").write_bytes(b"FAKE_T1")
    (base / "h2o.cc-pvdz.ccsd.t2_amp").write_bytes(b"FAKE_T2")
    # MO file
    (base / "h2o.cc-pvdz.movecs").write_bytes(b"FAKE_MO")
    # Cholesky vector
    (base / "h2o.cc-pvdz.cholvecs").write_bytes(b"FAKE_CHOL")
    # Output JSON that should NOT be staged
    (base / "h2o.cc-pvdz.scf.json").write_text("{}")
    # Log file that should NOT be staged
    (root / "exachem.log").write_text("done\n")


def test_classify_file_kind_recognizes_canonical_artifacts():
    assert _classify_file_kind("h2o.cc-pvdz.ccsd.t1_amp") == "amplitudes"
    assert _classify_file_kind("h2o.cc-pvdz.ccsd.t2_amp") == "amplitudes"
    assert _classify_file_kind("h2o.cc-pvdz.movecs") == "mo"
    assert _classify_file_kind("h2o.cc-pvdz.cholvecs") == "cholesky"
    assert _classify_file_kind("anything.restart") == "restart"
    assert _classify_file_kind("exachem.log") is None
    assert _classify_file_kind("input.json") is None


def test_resolve_restart_artifact_passes_through_existing_path(tmp_path):
    d = tmp_path / "prev_run"
    d.mkdir()
    assert resolve_restart_artifact(str(d)) == d
    assert resolve_restart_artifact(d) == d


def test_resolve_restart_artifact_missing_path_with_no_db_returns_none(caplog):
    # Looks like a hash but no DB available; should warn + return None.
    fake_hash = "a" * 64
    out = resolve_restart_artifact(fake_hash, db_path=None)
    assert out is None


def test_resolve_restart_artifact_returns_none_when_table_missing(tmp_path):
    """Hash-based lookup must not raise when F4 hasn't been merged."""
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    # Intentionally do NOT create the calculations table.
    conn.close()
    out = resolve_restart_artifact("a" * 64, db_path=db)
    assert out is None


def test_resolve_restart_artifact_finds_archive_via_blob_data(tmp_path):
    """Hash lookup should locate an archive recorded in blob_data."""
    archive = tmp_path / "fake.tar.gz"
    # Create a real tar.gz so Path.exists() succeeds.
    with tarfile.open(archive, "w:gz") as tf:
        f = tmp_path / "placeholder"
        f.write_text("x")
        tf.add(f, arcname="placeholder")
    db = tmp_path / "manifest.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE calculations ("
        "id INTEGER PRIMARY KEY, geometry_hash TEXT, params_hash TEXT, "
        "calculator TEXT, model TEXT, task TEXT, blob_data TEXT)"
    )
    blob = {
        "geometry_hash": "a" * 64,
        "artifact_archive": str(archive),
        "run_dir": None,
    }
    conn.execute(
        "INSERT INTO calculations VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "a" * 64, "p" * 64, "exachem", "ccsd_t/cc-pvdz", "energy", json.dumps(blob)),
    )
    conn.commit()
    conn.close()

    found = resolve_restart_artifact("a" * 64, db_path=db)
    assert found == archive


def test_stage_restart_artifacts_from_directory_copies_only_restartable(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_fake_artifact_files(src)
    dest = tmp_path / "new_run"

    staged = _stage_restart_artifacts(src, dest)

    names = sorted(p.name for p in staged)
    assert "h2o.cc-pvdz.ccsd.t1_amp" in names
    assert "h2o.cc-pvdz.ccsd.t2_amp" in names
    assert "h2o.cc-pvdz.movecs" in names
    assert "h2o.cc-pvdz.cholvecs" in names
    # Output JSON and logs must NOT be staged.
    assert not any("scf.json" in n for n in names)
    assert "exachem.log" not in names
    # Files are real copies (not zero-byte stubs).
    assert (dest / "h2o.cc-pvdz.ccsd.t1_amp").read_bytes() == b"FAKE_T1"


def test_stage_restart_artifacts_from_targz_extracts_and_copies(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_fake_artifact_files(src)
    archive = tmp_path / "prev.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname="src")

    dest = tmp_path / "new_run"
    staged = _stage_restart_artifacts(archive, dest)
    names = sorted(p.name for p in staged)
    assert "h2o.cc-pvdz.ccsd.t1_amp" in names
    assert "h2o.cc-pvdz.movecs" in names
    assert not any("scf.json" in n for n in names)


def test_stage_restart_artifacts_with_manifest_uses_kind_filter(tmp_path):
    """A F4-shaped manifest drives selection and ignores unlisted kinds."""
    src = tmp_path / "src"
    src.mkdir()
    _write_fake_artifact_files(src)

    base = src / "h2o.cc-pvdz_files" / "restricted"
    manifest = [
        {"path": str(base / "h2o.cc-pvdz.ccsd.t1_amp"), "kind": "amplitudes",
         "size_bytes": 7, "sha256": "x"},
        {"path": str(base / "h2o.cc-pvdz.movecs"), "kind": "mo",
         "size_bytes": 7, "sha256": "x"},
        # 'output' kind must be skipped.
        {"path": str(base / "h2o.cc-pvdz.scf.json"), "kind": "output",
         "size_bytes": 2, "sha256": "x"},
    ]
    dest = tmp_path / "new_run"
    staged = _stage_restart_artifacts(src, dest, manifest=manifest)
    names = sorted(p.name for p in staged)
    assert names == sorted(["h2o.cc-pvdz.ccsd.t1_amp", "h2o.cc-pvdz.movecs"])


def test_stage_restart_artifacts_missing_source_raises(tmp_path):
    from ase.calculators.calculator import CalculationFailed

    with pytest.raises(CalculationFailed):
        _stage_restart_artifacts(tmp_path / "nope", tmp_path / "run")


def test_build_input_json_strips_restart_from(water):
    """restart_from is an F10 hint and must not appear in the ExaChem JSON."""
    calc = ExaChemCalculator(
        method="ccsd",
        exachem_input={"restart_from": "/some/archive.tar.gz",
                       "SCF": {"writem": 5}},
    )
    payload = calc._build_input_json(water, calc.parameters, "ccsd")
    assert "restart_from" not in payload
    assert payload["SCF"]["writem"] == 5


def test_calculate_stages_amplitudes_before_mpiexec(tmp_path, water):
    """End-to-end: a CCSD restart_from archive stages T1/T2 + MO files
    into the new run_dir before subprocess.run is invoked, then short-
    circuits the actual ExaChem binary call.
    """
    # Build a fake archive of a prior run.
    prev = tmp_path / "prev"
    prev.mkdir()
    _write_fake_artifact_files(prev)
    archive = tmp_path / "prev.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(prev, arcname="prev")

    work = tmp_path / "work"
    work.mkdir()
    calc = ExaChemCalculator(
        method="ccsd",
        basis="cc-pvdz",
        directory=str(work),
        binary="/usr/bin/true",
        exachem_input={"restart_from": str(archive)},
    )
    water.calc = calc

    # Capture what subprocess.run sees AT call time and assert that the
    # amplitudes are already staged before mpiexec/ExaChem launches.
    staged_at_launch = {}

    def fake_run(cmd, cwd, stdout, stderr, env, check):
        run_dir = Path(cwd)
        staged_at_launch["names"] = sorted(p.name for p in run_dir.iterdir())
        # Drop a minimal SCF JSON so _locate_output succeeds and the
        # calculator doesn't raise CalculationFailed.
        out_dir = run_dir / "input.cc-pvdz_files" / "restricted" / "json"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "input.cc-pvdz.ccsd.json").write_text(
            json.dumps(
                {
                    "output": {
                        "SCF": {"final_energy": -76.0},
                        "CCSD": {
                            "final_energy": {"correlation": -0.2, "total": -76.2}
                        },
                    },
                    "input": {
                        "basis": {"basisset": "cc-pvdz"},
                        "SCF": {"scf_type": "restricted"},
                        "TASK": {"ccsd": True},
                    },
                }
            )
        )

        class _R:
            returncode = 0

        return _R()

    with mock.patch("iqc.exachem.subprocess.run", side_effect=fake_run):
        energy_eV = water.get_potential_energy()

    assert "h2o.cc-pvdz.ccsd.t1_amp" in staged_at_launch["names"]
    assert "h2o.cc-pvdz.ccsd.t2_amp" in staged_at_launch["names"]
    assert "h2o.cc-pvdz.movecs" in staged_at_launch["names"]
    # input.json must still be present alongside the staged artifacts.
    assert "input.json" in staged_at_launch["names"]
    assert energy_eV < 0


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


# --- F4: artifact manifest tests --------------------------------------------


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_fake_run_dir(root: Path) -> dict:
    """Populate ``root`` with the artifact types ExaChem writes; return paths."""

    stem = "h2o"
    basis = "cc-pvdz"
    nested = root / f"{stem}.{basis}_files" / "restricted"
    files = {
        "mo": nested / f"{stem}.{basis}.movecs",
        "molden": nested / f"{stem}.{basis}.molden",
        "t1": nested / f"{stem}.{basis}.t1amp",
        "t2": nested / f"{stem}.{basis}.t2amp",
        "chol": nested / f"{stem}.{basis}.cholesky.dat",
        "restart": nested / f"{stem}.{basis}.restart",
        "output": nested / "json" / f"{stem}.{basis}.ccsd_t.json",
        "log": root / "exachem.log",
        "stray": nested / "scratch.tmp",
    }
    # Deterministic contents so sha256 is reproducible.
    payloads = {
        name: f"contents-of-{name}".encode("utf-8") for name in files
    }
    for name, path in files.items():
        _write(path, payloads[name])
    return {"paths": files, "payloads": payloads}


def test_classify_artifact_kinds():
    classify = ExaChemCalculator._classify_artifact
    assert classify("h2o.cc-pvdz.movecs") == "mo"
    assert classify("h2o.molden") == "mo"
    assert classify("h2o.t1amp") == "amplitudes"
    assert classify("h2o.t2amp") == "amplitudes"
    assert classify("h2o.cholesky.dat") == "cholesky"
    assert classify("h2o.cd_vec.bin") == "cholesky"
    assert classify("h2o.restart") == "restart"
    assert classify("h2o.chkpt") == "restart"
    assert classify("h2o.ccsd.json") == "output"
    assert classify("exachem.log") == "output"
    assert classify("scratch.tmp") == "other"


def test_build_manifest_collects_expected_kinds(tmp_path):
    fake = _make_fake_run_dir(tmp_path)
    output_path = fake["paths"]["output"]

    manifest = ExaChemCalculator._build_artifact_manifest(tmp_path, output_path)

    by_path = {entry["path"]: entry for entry in manifest}
    # Every fake file (no symlinks) should appear.
    for path in fake["paths"].values():
        assert str(path.resolve()) in by_path, f"missing {path}"

    # Spot-check kinds for representative files.
    assert by_path[str(fake["paths"]["mo"].resolve())]["kind"] == "mo"
    assert by_path[str(fake["paths"]["t1"].resolve())]["kind"] == "amplitudes"
    assert by_path[str(fake["paths"]["t2"].resolve())]["kind"] == "amplitudes"
    assert by_path[str(fake["paths"]["chol"].resolve())]["kind"] == "cholesky"
    assert by_path[str(fake["paths"]["restart"].resolve())]["kind"] == "restart"
    # The known output JSON is tagged 'output' even though the JSON glob
    # rule would also match it.
    assert by_path[str(fake["paths"]["output"].resolve())]["kind"] == "output"
    assert by_path[str(fake["paths"]["stray"].resolve())]["kind"] == "other"


def test_build_manifest_paths_are_absolute(tmp_path):
    _make_fake_run_dir(tmp_path)
    # Pass a relative-looking path; resolve() inside the builder must promote
    # entries to absolute regardless.
    manifest = ExaChemCalculator._build_artifact_manifest(tmp_path)
    assert manifest, "expected non-empty manifest"
    for entry in manifest:
        assert os.path.isabs(entry["path"]), entry
        # And no '..' segments slipping through.
        assert ".." not in Path(entry["path"]).parts


def test_build_manifest_sha256_and_size(tmp_path):
    fake = _make_fake_run_dir(tmp_path)
    manifest = ExaChemCalculator._build_artifact_manifest(
        tmp_path, fake["paths"]["output"]
    )
    by_path = {entry["path"]: entry for entry in manifest}
    for name, path in fake["paths"].items():
        entry = by_path[str(path.resolve())]
        payload = fake["payloads"][name]
        assert entry["size_bytes"] == len(payload), (name, entry)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest(), (name, entry)
        # All values JSON-serializable scalars.
        assert isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64
        assert isinstance(entry["size_bytes"], int)


def test_build_manifest_skips_symlinks(tmp_path):
    """Symlinks must not produce duplicate entries pointing at the target."""

    real = tmp_path / "real.movecs"
    real.write_bytes(b"data")
    link = tmp_path / "link.movecs"
    link.symlink_to(real)
    manifest = ExaChemCalculator._build_artifact_manifest(tmp_path)
    # Exactly one entry — the real file — should appear; the symlink should
    # have been skipped during the walk to avoid double-hashing the same
    # bytes and emitting a stale alias path.
    assert len(manifest) == 1
    assert manifest[0]["path"] == str(real.resolve())


def test_build_manifest_empty_for_missing_dir(tmp_path):
    missing = tmp_path / "nope"
    assert ExaChemCalculator._build_artifact_manifest(missing) == []


def _fake_calculate(self, atoms=None, properties=("energy",), system_changes=None):
    """Stand-in for ExaChemCalculator.calculate that does not need a binary.

    We mimic the side effects the real method has on ``self.results`` and
    ``self.last_*`` so the manifest-building branch can run end-to-end.
    """

    from ase.calculators.calculator import Calculator as _Cal

    _Cal.calculate(self, atoms, list(properties), [])
    params = self.parameters
    keep_artifacts = bool(params.get("keep_artifacts", False))
    keep_files = bool(params.get("keep_files", False)) or keep_artifacts
    run_dir = self._prepare_run_dir(keep_files)
    # Drop fake artifacts into the prepared run_dir so the manifest sees
    # them when --keep-artifacts is on.
    fake = _make_fake_run_dir(run_dir)
    output_path = fake["paths"]["output"]

    self.last_run_dir = run_dir
    self.last_output_path = output_path
    self.last_output_payload = {"output": {"SCF": {"final_energy": -76.0}}}
    self.results = {
        "energy": -76.0 * units.Hartree,
        "energy_hartree": -76.0,
        "exachem_output": self.last_output_payload,
    }
    if keep_artifacts:
        manifest = self._build_artifact_manifest(run_dir, output_path)
        self.results["run_dir"] = str(run_dir.resolve())
        self.results["artifact_manifest"] = manifest
        self.results["keep_artifacts"] = True
    else:
        self.results["run_dir"] = None
        self.results["artifact_manifest"] = []
        self.results["keep_artifacts"] = False
    self.results["artifact_archive"] = None


def test_calculate_records_run_dir_and_manifest_when_keep_artifacts(tmp_path, water):
    calc = ExaChemCalculator(
        method="scf",
        directory=str(tmp_path),
        keep_artifacts=True,
    )
    water.calc = calc
    with mock.patch.object(ExaChemCalculator, "calculate", _fake_calculate):
        calc.calculate(water)

    assert calc.results["keep_artifacts"] is True
    assert calc.results["artifact_archive"] is None
    assert calc.results["run_dir"] is not None
    assert os.path.isabs(calc.results["run_dir"])
    assert Path(calc.results["run_dir"]).is_dir()
    manifest = calc.results["artifact_manifest"]
    assert isinstance(manifest, list) and len(manifest) > 0
    kinds = {entry["kind"] for entry in manifest}
    # MO, amplitudes, cholesky, restart, output were all written by the fake.
    for required in ("mo", "amplitudes", "cholesky", "restart", "output"):
        assert required in kinds, kinds


def test_calculate_default_leaves_manifest_empty_and_run_dir_null(tmp_path, water):
    calc = ExaChemCalculator(method="scf", directory=str(tmp_path))
    water.calc = calc
    with mock.patch.object(ExaChemCalculator, "calculate", _fake_calculate):
        calc.calculate(water)

    assert calc.results["keep_artifacts"] is False
    assert calc.results["run_dir"] is None
    assert calc.results["artifact_manifest"] == []
    assert calc.results["artifact_archive"] is None


def test_keep_artifacts_false_reclaims_previous_scratch(tmp_path, water):
    """With keep_artifacts/keep_files off, each call gets a FRESH unique run
    dir (so concurrent/serial reuse never collides on a fixed name), and the
    previous call's scratch is reclaimed on the next call so it does not
    accumulate.
    """

    calc = ExaChemCalculator(method="scf", directory=str(tmp_path))
    water.calc = calc
    with mock.patch.object(ExaChemCalculator, "calculate", _fake_calculate):
        calc.calculate(water)
        first_run_dir = calc.last_run_dir
        sentinel = first_run_dir / "sentinel.dat"
        sentinel.write_text("from first run")
        assert sentinel.exists()

        calc.calculate(water)
        # Each call gets its own unique dir (Parsl-safe)...
        assert calc.last_run_dir != first_run_dir
        # ...and the previous non-kept scratch (incl. the sentinel) is reclaimed.
        assert not first_run_dir.exists()
        assert not sentinel.exists()
        assert calc.last_run_dir.exists()
        assert calc.last_run_dir.parent == first_run_dir.parent


def test_keep_artifacts_true_preserves_run_dir_between_calls(tmp_path, water):
    calc = ExaChemCalculator(method="scf", directory=str(tmp_path), keep_artifacts=True)
    water.calc = calc
    with mock.patch.object(ExaChemCalculator, "calculate", _fake_calculate):
        calc.calculate(water)
        first_run_dir = calc.last_run_dir
        sentinel = first_run_dir / "sentinel.dat"
        sentinel.write_text("from first run")

        calc.calculate(water)
        assert sentinel.exists(), "keep_artifacts should imply keep_files"


def test_store_calculator_observables_copies_artifact_fields(tmp_path, water):
    calc = ExaChemCalculator(
        method="scf",
        directory=str(tmp_path),
        keep_artifacts=True,
    )
    water.calc = calc
    with mock.patch.object(ExaChemCalculator, "calculate", _fake_calculate):
        calc.calculate(water)

    out = _store_calculator_observables({}, calc)
    assert out["run_dir"] == calc.results["run_dir"]
    assert out["artifact_manifest"] == calc.results["artifact_manifest"]
    assert out["keep_artifacts"] is True
    assert out["artifact_archive"] is None
    # And no path leaks through as relative.
    for entry in out["artifact_manifest"]:
        assert os.path.isabs(entry["path"])


def test_store_calculator_observables_prefix_applies_to_artifact_fields(tmp_path, water):
    calc = ExaChemCalculator(
        method="scf",
        directory=str(tmp_path),
        keep_artifacts=True,
    )
    water.calc = calc
    with mock.patch.object(ExaChemCalculator, "calculate", _fake_calculate):
        calc.calculate(water)

    out = _store_calculator_observables({}, calc, prefix="initial_")
    assert "initial_run_dir" in out
    assert "initial_artifact_manifest" in out
    assert "initial_keep_artifacts" in out
    assert "initial_artifact_archive" in out


def test_keep_artifacts_default_parameter_exists():
    """Pin the calculator's default so callers can rely on backward-compat."""

    assert ExaChemCalculator.default_parameters["keep_artifacts"] is False
