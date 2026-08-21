"""Tests for rdkittools module."""

import unittest
from unittest.mock import patch
from rdkit import Chem
from iqc.rdkittools import (
    xyz_to_mol,
    check_mol,
    get_formula,
    get_inchi,
    get_smiles,
    get_inchikey,
    get_mol_weight,
    get_num_atoms,
    get_num_bonds,
    get_num_heavy_atoms,
    get_num_hydrogens,
    get_bond_order,
    get_bond_orders,
    get_max_bond_order,
    get_num_electrons,
    get_num_radical_electrons,
)


class TestRDKitTools(unittest.TestCase):
    """Tests for rdkittools module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test molecule - methane
        self.methane_xyz = """5
Methane
C       0.000000    0.000000    0.000000
H       0.000000    0.000000    1.089000
H       1.026719    0.000000   -0.363000
H      -0.513360   -0.889165   -0.363000
H      -0.513360    0.889165   -0.363000"""
        self.methane_mol = xyz_to_mol(self.methane_xyz)

        # Create another test molecule - water
        self.water_xyz = """3
Water
O       0.000000    0.000000    0.000000
H       0.000000    0.000000    0.950000
H       0.950000    0.000000    0.000000"""
        self.water_mol = xyz_to_mol(self.water_xyz)

    def test_xyz_to_mol(self):
        """Test xyz_to_mol function."""
        # Test with methane
        mol = xyz_to_mol(self.methane_xyz)
        self.assertIsNotNone(mol)
        self.assertEqual(get_formula(mol), "CH4")

        # Test with water
        mol = xyz_to_mol(self.water_xyz)
        self.assertIsNotNone(mol)
        self.assertEqual(get_formula(mol), "H2O")

        # Test with invalid xyz
        with patch("builtins.print"):  # Suppress print statements
            mol = xyz_to_mol("invalid xyz")
            self.assertIsNone(mol)

    def test_check_mol(self):
        """Test check_mol function."""
        # Valid molecule
        with patch("builtins.print"):  # Suppress print statements
            problems = check_mol(self.methane_mol)
            self.assertEqual(len(problems), 0)

    def test_get_formula(self):
        """Test get_formula function."""
        self.assertEqual(get_formula(self.methane_mol), "CH4")
        self.assertEqual(get_formula(self.water_mol), "H2O")

    def test_get_inchi(self):
        """Test get_inchi function."""
        # InChI string format can vary, so we'll just check it's not empty
        self.assertTrue(get_inchi(self.methane_mol).startswith("InChI="))
        self.assertTrue(get_inchi(self.water_mol).startswith("InChI="))

    def test_get_smiles(self):
        """Test get_smiles function."""
        # Test with methane
        smiles = get_smiles(self.methane_mol)
        self.assertEqual(smiles, "C")  # Methane is just "C" in SMILES

        # Test with water
        smiles = get_smiles(self.water_mol)
        self.assertEqual(smiles, "O")  # Water without explicit hydrogens is "O"

        # Test with explicit hydrogens - accept the actual format RDKit produces
        smiles = get_smiles(self.methane_mol, remove_hydrogens=False)
        self.assertIn("C", smiles)  # Should include carbon
        self.assertIn("H", smiles)  # Should include hydrogen
        self.assertEqual(smiles.count("H"), 4)  # Should have 4 hydrogens

    def test_get_inchikey(self):
        """Test get_inchikey function."""
        # INCHIKEY is 27 characters in a specific format
        methane_key = get_inchikey(self.methane_mol)
        self.assertEqual(len(methane_key), 27)
        self.assertTrue("-" in methane_key)

        water_key = get_inchikey(self.water_mol)
        self.assertEqual(len(water_key), 27)
        self.assertTrue("-" in water_key)

    def test_get_mol_weight(self):
        """Test get_mol_weight function."""
        # Exact molecular weights
        self.assertAlmostEqual(get_mol_weight(self.methane_mol), 16.03, places=2)
        self.assertAlmostEqual(get_mol_weight(self.water_mol), 18.01, places=2)

    def test_get_num_atoms(self):
        """Test get_num_atoms function."""
        self.assertEqual(get_num_atoms(self.methane_mol), 5)  # C + 4H
        self.assertEqual(get_num_atoms(self.water_mol), 3)  # O + 2H

    def test_get_num_bonds(self):
        """Test get_num_bonds function."""
        self.assertEqual(get_num_bonds(self.methane_mol), 4)  # 4 C-H bonds
        self.assertEqual(get_num_bonds(self.water_mol), 2)  # 2 O-H bonds

    def test_get_num_heavy_atoms(self):
        """Test get_num_heavy_atoms function."""
        self.assertEqual(get_num_heavy_atoms(self.methane_mol), 1)  # 1 C
        self.assertEqual(get_num_heavy_atoms(self.water_mol), 1)  # 1 O

    def test_get_num_hydrogens(self):
        """Hydrogen counts must be actual H atoms, not H-bond donors."""
        # Methane has 4 hydrogens (the old CalcNumHBD implementation said 0).
        self.assertEqual(get_num_hydrogens(self.methane_mol), 4)
        self.assertEqual(get_num_hydrogens(self.water_mol), 2)

        # Implicit hydrogens (SMILES without AddHs) are counted too.
        ethylene = Chem.MolFromSmiles("C=C")
        self.assertEqual(get_num_hydrogens(ethylene), 4)

    def test_get_max_bond_order(self):
        """Bond order must come from bonds, not atom degrees."""
        # Methane's max bond order is 1.0 (the old degree-based code said 4).
        self.assertEqual(get_max_bond_order(self.methane_mol), 1.0)
        self.assertEqual(get_max_bond_order(self.water_mol), 1.0)

        ethylene = Chem.MolFromSmiles("C=C")
        self.assertEqual(get_max_bond_order(ethylene), 2.0)

        # An atom-free molecule has no bonds.
        self.assertEqual(get_max_bond_order(Chem.Mol()), 0.0)

    def test_get_bond_orders_returns_orders_for_atom(self):
        """Per-atom bond orders, not the atom degree."""
        ethylene = Chem.MolFromSmiles("C=C")
        self.assertEqual(get_bond_orders(ethylene, 0), [2.0])

    def test_smiles_to_mol_invalid_smiles_returns_none(self):
        """Invalid SMILES must return None instead of crashing in AddHs."""
        from iqc.rdkittools import smiles_to_mol

        self.assertIsNone(smiles_to_mol("not_a_smiles(("))

    def test_get_num_electrons(self):
        """Test get_num_electrons function."""
        # Methane: C(6) + 4*H(1) = 10 electrons
        electrons = get_num_electrons(self.methane_mol)
        self.assertEqual(electrons, 10)

        # Water: O(8) + 2*H(1) = 10 electrons
        electrons = get_num_electrons(self.water_mol)
        self.assertEqual(electrons, 10)

    def test_get_num_radical_electrons(self):
        """Test get_num_radical_electrons function."""
        # Neither methane nor water have radical electrons in their ground state
        self.assertEqual(get_num_radical_electrons(self.methane_mol), 0)
        self.assertEqual(get_num_radical_electrons(self.water_mol), 0)


if __name__ == "__main__":
    unittest.main()
