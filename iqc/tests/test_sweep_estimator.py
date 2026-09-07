"""Tests for iqc.sweep_estimator helpers."""

from iqc.sweep_estimator import _heavy_from_formula


def test_heavy_from_formula_skips_hydrogen():
    assert _heavy_from_formula("C2H6") == 2
    assert _heavy_from_formula("H2O") == 1


def test_heavy_from_formula_counts_h_prefixed_elements():
    """The old regex excluded H as a first letter, dropping He/Hf/Hg/Ho/Hs."""
    assert _heavy_from_formula("C2H6Hg") == 3
    assert _heavy_from_formula("He") == 1
    assert _heavy_from_formula("HfO2") == 3


def test_heavy_from_formula_empty_returns_none():
    assert _heavy_from_formula("") is None
    assert _heavy_from_formula("H2") is None
