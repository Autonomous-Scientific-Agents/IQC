"""Tests for nonphysical-result validation, no-op detection, and optimizer
selection added to harden IQC against silent-calculator and blow-up failures.

Motivated by the ROSMI runs where (a) UMA-on-XPU silently fell back to MACE and
produced 0-step "results", and (b) the mace-polar model diverged during BFGS,
writing energies up to ~1e56 eV that nothing flagged.
"""

import numpy as np
import pytest

from iqc.asetools import (
    validate_physical_results,
    check_suspicious_no_op,
    _make_optimizer,
    get_calculator,
)


def test_validate_clean_result_not_flagged():
    r = {
        "number_of_atoms": 10,
        "opt_energy_eV": -266.0,
        "G_eV": -259.0,
        "H_eV": -256.0,
        "E_ZPE_eV": 7.5,
        "S_eV/K": 0.008,
        "vibrational_frequencies_cm^-1": [100.0, 200.0, 3000.0],
        "warnings": [],
    }
    validate_physical_results(r)
    assert r["nonphysical"] is False
    assert not r.get("validation_messages")


def test_validate_blowup_energy_and_zpe():
    r = {
        "number_of_atoms": 50,
        "opt_energy_eV": 3.6e56,
        "E_ZPE_eV": 1e31,
        "warnings": [],
    }
    validate_physical_results(r)
    assert r["nonphysical"] is True
    assert any("opt_energy_eV" in m for m in r["validation_messages"])
    assert any("E_ZPE_eV" in m for m in r["validation_messages"])
    # messages are mirrored into warnings for downstream visibility
    assert any("opt_energy_eV" in w for w in r["warnings"])


def test_validate_negative_entropy_and_nan_and_bad_freq():
    r = {
        "number_of_atoms": 5,
        "S_eV/K": -0.01,
        "G_eV": float("nan"),
        "vibrational_frequencies_cm^-1": [1e28, float("inf")],
        "warnings": [],
    }
    validate_physical_results(r)
    assert r["nonphysical"] is True
    joined = " ".join(r["validation_messages"])
    assert "S_eV/K" in joined and "negative" in joined
    assert "G_eV" in joined and "non-finite" in joined
    assert "frequenc" in joined


def test_validate_never_raises_and_is_idempotent():
    r = {"number_of_atoms": 3, "opt_energy_eV": 1e60, "warnings": []}
    validate_physical_results(r)
    n1 = len(r["validation_messages"])
    # second call must not duplicate messages or raise
    validate_physical_results(r)
    assert len(r["validation_messages"]) == n1
    assert r["nonphysical"] is True


def test_validate_imaginary_is_informational_not_nonphysical():
    r = {"number_of_atoms": 5, "number_of_imaginary": 3, "G_eV": -10.0, "warnings": []}
    validate_physical_results(r)
    assert r.get("has_imaginary") is True
    assert r["nonphysical"] is False


def test_no_op_optimization_flagged():
    r = {
        "opt_steps": 0,
        "opt_time": 0.008,
        "initial_energy_eV": -544.37783,
        "opt_energy_eV": -544.37783,
        "warnings": [],
    }
    check_suspicious_no_op(r)
    assert r.get("suspicious_no_op") is True
    assert any("no-op" in w for w in r["warnings"])


def test_slow_zero_step_optimization_not_flagged():
    # A genuinely pre-converged geometry (0 steps but real wall time) is legal.
    r = {
        "opt_steps": 0,
        "opt_time": 5.0,
        "initial_energy_eV": -1.0,
        "opt_energy_eV": -1.0,
        "warnings": [],
    }
    check_suspicious_no_op(r)
    assert "suspicious_no_op" not in r


def test_no_op_guard_ignores_non_optimization_results():
    r = {"energy_eV": -1.0, "warnings": []}  # single-point: no opt_steps
    check_suspicious_no_op(r)
    assert "suspicious_no_op" not in r


def test_make_optimizer_selection():
    from ase.build import molecule
    from ase.optimize import BFGS, FIRE, LBFGS

    atoms = molecule("H2O")
    assert isinstance(_make_optimizer("bfgs", atoms), BFGS)
    assert isinstance(_make_optimizer("lbfgs", atoms), LBFGS)
    assert isinstance(_make_optimizer("fire", atoms), FIRE)
    with pytest.raises(ValueError):
        _make_optimizer("nope", atoms)


def test_unknown_calculator_raises():
    # Requirement #1: an unavailable/unknown calculator must error, never
    # silently substitute a different level of theory.
    with pytest.raises(RuntimeError):
        get_calculator("definitely-not-a-real-calculator")


def test_run_optimization_coerces_string_fmax():
    # YAML 1.1 parses "1e-3" as a str; run_optimization must coerce numeric
    # params so a config fmax never triggers a cryptic ufunc type error.
    from ase.build import molecule
    from ase.calculators.emt import EMT
    from iqc.asetools import run_optimization

    atoms = molecule("H2O")
    atoms.rattle(0.05, seed=1)
    _, res = run_optimization(
        atoms, calculator=EMT(), fmax="1e-3", max_steps="200", maxstep="0.2",
        unique_name="h2o_strfmax",
    )
    assert not res.get("error")
    assert res["opt_converged"] is True
