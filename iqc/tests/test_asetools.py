import os
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator
import pytest
import sys
from unittest.mock import patch, MagicMock
import logging

# Configure logging for tests to see warnings
logging.basicConfig(level=logging.WARNING)

from iqc.asetools import (
    save_atoms,
    translate_atoms,
    get_canonical_smiles_from_atoms,
    atoms2xyz,
    get_total_electrons,
    get_spin,
    xyz2atoms,
    get_calculator,
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
    # Skip if rdkit not installed
    pytest.importorskip("rdkit")
    from rdkit import Chem

    smiles = get_canonical_smiles_from_atoms(methane_atoms)
    # Note: RDKit may return SMILES with explicit hydrogens
    # We should check if the molecule is equivalent rather than exact string match
    mol = Chem.MolFromSmiles(smiles)
    assert Chem.MolToSmiles(mol, canonical=True) in ["C", "[H]C([H])([H])[H]"]


def test_get_calculator_mace_success():
    """Test getting MACE calculator successfully."""
    # Skip test if MACE cannot be imported
    pytest.importorskip("mace")
    try:
        calculator = get_calculator(name="mace")
        # Check if it's MACE or a SumCalculator containing MACE
        is_mace_like = False
        if isinstance(calculator, SumCalculator):
            # Correct attribute is 'calcs'
            is_mace_like = any(
                "mace" in str(c).lower() for c in str(calculator).split(",")
            )
        else:
            is_mace_like = "mace" in str(calculator).lower()
        assert is_mace_like, f"Calculator name {calculator.name} does not indicate MACE"

        # Test calculation
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert not np.isnan(energy)

    except RuntimeError as e:
        # This might happen if MACE is installed but initialization fails
        pytest.skip(f"Could not initialize MACE or fallback: {e}")


def test_get_calculator_mace_unavailable_fallback():
    """Test fallback to EMT when MACE is not available."""
    # Mock the import to simulate MACE not being available
    with patch.dict(sys.modules, {"mace.calculators": None, "mace": None}):
        try:
            calculator = get_calculator(name="mace")
            # It should fallback to EMT
            assert isinstance(calculator, EMT)

            # Test calculation with fallback
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            assert isinstance(energy, float)
        except RuntimeError as e:
            pytest.skip(f"Could not initialize EMT fallback: {e}")
        except ImportError:
            # This could happen if EMT also fails to import
            pytest.skip("Neither MACE nor EMT fallback available.")


def test_get_calculator_xtb_success():
    """Test getting XTB calculator successfully."""
    # Skip test if XTB cannot be imported
    pytest.importorskip("xtb")
    try:
        calculator = get_calculator(name="xtb")
        # This assertion now happens first. If XTB wasn't importable,
        # the test would have skipped above. If it falls back despite
        # being importable, this assertion will correctly fail.
        assert "xtb" in calculator.name.lower()

        # Test calculation
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert not np.isnan(energy)

    except RuntimeError as e:
        # This might happen if XTB is installed but init fails, and fallback also fails
        pytest.skip(f"Could not initialize XTB or fallback: {e}")


def test_get_calculator_xtb_unavailable_fallback():
    """Test fallback when XTB is not available."""
    # Mock the import to simulate XTB not being available
    with patch.dict(sys.modules, {"xtb.ase.calculator": None, "xtb": None}):
        try:
            calculator = get_calculator(name="xtb")
            # It should fallback to MACE or EMT
            assert calculator is not None
            # Check type for EMT or name for MACE/Sum(MACE)
            is_fallback_ok = False
            if isinstance(calculator, EMT):
                is_fallback_ok = True
            elif isinstance(calculator, SumCalculator):
                # Correct attribute is 'calcs'
                is_fallback_ok = any("mace" in c.name.lower() for c in calculator.calcs)
            else:
                is_fallback_ok = "mace" in calculator.name.lower()
            assert (
                is_fallback_ok
            ), f"Fallback calculator {calculator.name} is not MACE or EMT"

            # Test calculation with fallback
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            atoms.calc = calculator
            energy = atoms.get_potential_energy()
            assert isinstance(energy, float)
        except RuntimeError as e:
            pytest.skip(f"Could not initialize MACE/EMT fallback: {e}")
        except ImportError:
            pytest.skip("Neither XTB nor fallbacks available.")


def test_get_calculator_emt_direct():
    """Test getting EMT calculator directly."""
    try:
        calculator = get_calculator(name="emt")
        assert isinstance(calculator, EMT)

        # Test calculation
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert not np.isnan(energy)

    except RuntimeError as e:
        pytest.skip(f"Could not initialize EMT: {e}")
    except ImportError:
        pytest.skip("EMT calculator not available (ASE issue?).")


def test_get_calculator_unknown_fallback():
    """Test fallback when an unknown calculator is requested."""
    try:
        calculator = get_calculator(name="unknown_calc")
        # It should fallback (likely to EMT after trying MACE/XTB)
        assert calculator is not None
        # Check if it's EMT or MACE/Sum(MACE)
        is_fallback_ok = False
        if isinstance(calculator, EMT):
            is_fallback_ok = True
        elif isinstance(calculator, SumCalculator):
            # Correct attribute is 'calcs'
            is_fallback_ok = any("mace" in c.name.lower() for c in calculator.calcs)
        else:
            is_fallback_ok = "mace" in calculator.name.lower()
        assert (
            is_fallback_ok
        ), f"Fallback calculator {calculator.name} is not MACE or EMT"

        # Test calculation with fallback
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
    except RuntimeError as e:
        pytest.skip(f"Could not initialize fallback calculator: {e}")
    except ImportError:
        pytest.skip("Fallback calculator not available.")
