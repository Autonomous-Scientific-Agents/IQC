"""Tests for the polymer relaxation helper scripts."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from ase.build import molecule

from iqc.relax_polymer_ase import fix_overlapping_atoms
from iqc import packmoltools


@pytest.mark.parametrize("first", ["energy", "forces"])
def test_xtb_wrapper_refreshes_all_properties_on_movement(monkeypatch, first):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from iqc import relax_polymer_ase as polymer

    class FakeXTB(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": float((atoms.positions ** 2).sum()), "forces": -2 * atoms.positions.copy()}

    monkeypatch.setattr(polymer, "HAS_XTB", True)
    monkeypatch.setattr(polymer, "XTB", FakeXTB)
    atoms = Atoms("H", positions=[[1, 0, 0]], calculator=polymer.XTBForceOnly())
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.positions[0, 0] = 2
    if first == "energy":
        atoms.get_potential_energy()
    else:
        atoms.get_forces()
    assert atoms.get_potential_energy() == pytest.approx(4)
    assert np.allclose(atoms.get_forces(), [[-4, 0, 0]])


def test_polymer_npt_takes_real_ase_step(tmp_path, monkeypatch):
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from iqc import relax_polymer as polymer

    atoms = bulk("Cu", cubic=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(polymer.aio, "read", lambda _: atoms)
    monkeypatch.setattr(polymer, "build_calc", lambda *args: EMT())
    monkeypatch.setattr(polymer, "NVT_TIME", 0.001)
    monkeypatch.setattr(polymer, "NPT_TIME", 0.001)
    densities = iter([0.5, 1.0])
    monkeypatch.setattr(polymer, "density_g_cm3", lambda _: next(densities))
    monkeypatch.setattr("sys.argv", ["relax", "copper.xyz", "--ff", "xtb", "--scale", "1"])
    polymer.main()
    assert (tmp_path / "copper_relaxed.xyz").exists()


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
