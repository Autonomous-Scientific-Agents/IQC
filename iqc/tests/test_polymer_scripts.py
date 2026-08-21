"""Tests for the polymer relaxation helper scripts."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from ase.build import molecule

from iqc.relax_polymer_ase import fix_overlapping_atoms
from iqc import packmoltools


def test_fix_overlapping_atoms_default_keeps_bonds_intact():
    """The default 0.8 A cutoff must not touch normal covalent bonds.

    (The old call site passed min_distance=2.5, which pushed every bonded
    pair — C-H 1.1 A, O-H 0.96 A — out to 2.5 A, shredding the molecule.)
    """
    water = molecule("H2O")
    water.set_cell([10, 10, 10])
    water.center()
    before = water.get_positions().copy()

    fix_overlapping_atoms(water)

    assert np.allclose(water.get_positions(), before)


def test_fix_overlapping_atoms_separates_fused_atoms():
    from ase import Atoms

    fused = Atoms("HH", positions=[[0, 0, 0], [0.2, 0, 0]])
    fused.set_cell([10, 10, 10])
    fused.center()

    fix_overlapping_atoms(fused, min_distance=0.8)

    distance = np.linalg.norm(
        fused.get_positions()[0] - fused.get_positions()[1]
    )
    assert distance >= 0.75  # separated to ~min_distance


def test_run_packmol_feeds_input_on_stdin(tmp_path):
    """PACKMOL reads its input from stdin; the file must not be an argv."""
    inp = tmp_path / "pack.inp"
    inp.write_text("tolerance 2.0\n")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["stdin"] = kwargs.get("stdin")
        return mock.Mock(returncode=0, stdout="ok", stderr="")

    with mock.patch.object(packmoltools.subprocess, "run", fake_run):
        packmoltools.run_packmol(inp)

    assert captured["cmd"] == ["packmol"]
    assert captured["stdin"] is not None
    assert captured["stdin"].name == str(inp)
