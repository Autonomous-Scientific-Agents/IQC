import os
import numpy as np
from ase import Atoms
import pytest
import sys
from unittest.mock import patch, MagicMock

from iqc.asetools import (
    save_atoms,
    translate_atoms,
    get_canonical_smiles_from_atoms,
    atoms2xyz,
    get_total_electrons,
    get_spin,
    xyz2atoms,
    default_calculator,
    XTB,
)


# Test fixtures
@pytest.fixture
def water_atoms():
    """Create a simple water molecule for testing."""
    return Atoms(
        "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=False
    )


@pytest.fixture
def methane_atoms():
    """Create a methane molecule for testing."""
    return Atoms(
        "CH4",
        positions=[
            [0.0, 0.0, 0.0],  # C
            [0.6, 0.6, 0.6],  # H
            [-0.6, -0.6, 0.6],  # H
            [0.6, -0.6, -0.6],  # H
            [-0.6, 0.6, -0.6],  # H
        ],
        cell=[10, 10, 10],
        pbc=False,
    )


def test_save_atoms(water_atoms, tmp_path):
    """Test saving atoms to file."""
    file_path = save_atoms(water_atoms, prefix="test", directory=str(tmp_path))
    assert os.path.exists(file_path)
    assert file_path.endswith(".xyz")
    assert "test" in file_path
    assert "H2O" in file_path


def test_translate_atoms(water_atoms):
    """Test translating atoms."""
    # Translate H atoms in the direction of O atom
    translated = translate_atoms(
        water_atoms.copy(),
        indices=[0, 1],  # H atoms
        reference_index=2,  # O atom
        target_index=0,  # First H atom
        distance=1.0,
    )

    # Check that atoms were moved
    assert not np.allclose(water_atoms.positions, translated.positions)
    assert len(translated) == len(water_atoms)


def test_atoms2xyz(water_atoms):
    """Test converting atoms to XYZ format."""
    xyz_str = atoms2xyz(water_atoms)

    # Check basic XYZ format requirements
    lines = xyz_str.strip().split("\n")
    assert len(lines) == 5  # Number of atoms + 2 header lines
    assert lines[0].strip() == "3"  # Number of atoms
    assert lines[1].strip() == ""  # Comment line

    # Check atom lines format
    for line in lines[2:]:
        parts = line.split()
        assert len(parts) == 4  # Symbol and 3 coordinates
        # Check if coordinates can be converted to float
        assert all(isinstance(float(x), float) for x in parts[1:])


def test_get_total_electrons(water_atoms, methane_atoms):
    """Test electron counting."""
    # H2O: O(8) + 2*H(1) = 10 electrons
    assert get_total_electrons(water_atoms) == 10

    # CH4: C(6) + 4*H(1) = 10 electrons
    assert get_total_electrons(methane_atoms) == 10


def test_get_spin(water_atoms, methane_atoms):
    """Test spin calculation."""
    # Both water and methane have even number of electrons -> spin 0
    assert get_spin(water_atoms) == 0.0
    assert get_spin(methane_atoms) == 0.0

    # Create OH radical (odd number of electrons)
    oh_radical = Atoms("OH", positions=[[0, 0, 0], [0, 0, 1]])
    assert get_spin(oh_radical) == 0.5


def test_xyz2atoms():
    """Test converting XYZ string to atoms."""
    xyz_str = """3

O 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 1.0 0.0"""

    atoms = xyz2atoms(xyz_str)
    assert len(atoms) == 3
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert atoms.positions.shape == (3, 3)


def test_get_canonical_smiles(methane_atoms):
    """Test SMILES generation."""
    smiles = get_canonical_smiles_from_atoms(methane_atoms)
    # Note: RDKit may return SMILES with explicit hydrogens
    # We should check if the molecule is equivalent rather than exact string match
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    assert Chem.MolToSmiles(mol, canonical=True) in ["C", "[H]C([H])([H])[H]"]


def test_mace_calculator_available():
    """Test MACE calculator when available."""
    # Only run this test if MACE is actually available
    if default_calculator is not None:
        # Create a simple molecule
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

        # Set the calculator
        atoms.calc = default_calculator

        # Calculate energy
        energy = atoms.get_potential_energy()

        # Basic checks
        assert isinstance(energy, float)
        assert not np.isnan(energy)
    else:
        pytest.skip("MACE calculator not available")


def test_mace_calculator_unavailable():
    """Test behavior when MACE calculator is not available."""
    # Mock the import to simulate MACE not being available
    with patch.dict(sys.modules, {"mace.calculators": None}):
        # Re-import the module to get the updated default_calculator
        import importlib
        import iqc.asetools

        importlib.reload(iqc.asetools)

        # Check that default_calculator is None
        assert iqc.asetools.default_calculator is None

        # Create a simple molecule
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

        # Try to calculate energy without a calculator
        with pytest.raises(RuntimeError, match="Atoms object has no calculator"):
            atoms.get_potential_energy()


def test_xtb_calculator_available():
    """Test XTB calculator when available."""
    # Only run this test if XTB is actually available
    if XTB is not None:
        # Create a simple molecule
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

        # Set the calculator
        atoms.calc = XTB()

        # Calculate energy
        energy = atoms.get_potential_energy()

        # Basic checks
        assert isinstance(energy, float)
        assert not np.isnan(energy)
    else:
        pytest.skip("XTB calculator not available")


def test_xtb_calculator_unavailable():
    """Test behavior when XTB calculator is not available."""
    # Mock the import to simulate XTB not being available
    with patch.dict(sys.modules, {"xtb.ase.calculator": None}):
        # Re-import the module to get the updated XTB
        import importlib
        import iqc.asetools

        importlib.reload(iqc.asetools)

        # Check that XTB is None
        assert iqc.asetools.XTB is None

        # Create a simple molecule
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

        # Try to calculate energy without a calculator
        with pytest.raises(RuntimeError, match="Atoms object has no calculator"):
            atoms.get_potential_energy()
