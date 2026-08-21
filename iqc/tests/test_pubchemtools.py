"""Unit tests for pubchemtools module."""

import unittest
from unittest.mock import patch, MagicMock

from iqc import pubchemtools
from iqc.pubchemtools import (
    get_compound_from_cid,
    get_compounds_from_cids,
    get_pubchem_properties,
    get_pubchem_properties_batch,
    get_iupac_name,
    get_iupac_names,
    get_cid_from_inchi,
    get_cids_from_inchi,
    get_cid_from_smiles,
    get_cids_from_smiles,
)


class TestPubChemTools(unittest.TestCase):
    """Test the PubChem tools module."""

    @patch("iqc.pubchemtools.Compound.from_cid")
    def test_get_compound_from_cid(self, mock_from_cid):
        """Test getting a compound from CID."""
        # Setup mock
        mock_compound = MagicMock()
        mock_compound.molecular_formula = "H2O"
        mock_from_cid.return_value = mock_compound

        # Test function
        result = get_compound_from_cid("962")

        # Verify results
        mock_from_cid.assert_called_once_with("962")
        self.assertEqual(result, mock_compound)

    @patch("iqc.pubchemtools.get_compound_from_cid")
    def test_get_compounds_from_cids(self, mock_get_compound):
        """Test getting multiple compounds from CIDs."""
        # Setup mocks
        mock_compound1 = MagicMock()
        mock_compound1.molecular_formula = "H2O"
        mock_compound2 = MagicMock()
        mock_compound2.molecular_formula = "C2H6O"
        mock_get_compound.side_effect = [mock_compound1, mock_compound2]

        # Test function
        result = get_compounds_from_cids(["962", "702"])

        # Verify results
        self.assertEqual(mock_get_compound.call_count, 2)
        self.assertEqual(result, [mock_compound1, mock_compound2])

    @patch("iqc.pubchemtools.get_compound_from_cid")
    def test_get_pubchem_properties(self, mock_get_compound):
        """Test getting properties from a compound."""
        # Setup mock
        mock_compound = MagicMock()
        mock_compound.molecular_formula = "H2O"
        mock_compound.molecular_weight = 18.02
        mock_get_compound.return_value = mock_compound

        # Test function
        result = get_pubchem_properties("962")

        # Verify results
        mock_get_compound.assert_called_once_with("962")
        self.assertIn("molecular_formula", result)
        self.assertEqual(result["molecular_formula"], "H2O")
        self.assertIn("molecular_weight", result)
        self.assertEqual(result["molecular_weight"], 18.02)

    @patch("iqc.pubchemtools.get_pubchem_properties")
    def test_get_pubchem_properties_batch(self, mock_get_properties):
        """Test getting properties for multiple compounds."""
        # Setup mocks
        props1 = {"molecular_formula": "H2O", "molecular_weight": 18.02}
        props2 = {"molecular_formula": "C2H6O", "molecular_weight": 46.07}
        mock_get_properties.side_effect = [props1, props2]

        # Test function
        result = get_pubchem_properties_batch(["962", "702"])

        # Verify results
        self.assertEqual(mock_get_properties.call_count, 2)
        self.assertEqual(result, [props1, props2])

    @patch("iqc.pubchemtools.get_compound_from_cid")
    def test_get_iupac_name(self, mock_get_compound):
        """Test getting IUPAC name from a compound."""
        # Setup mock
        mock_compound = MagicMock()
        mock_compound.iupac_name = "water"
        mock_get_compound.return_value = mock_compound

        # Test function
        result = get_iupac_name("962")

        # Verify results
        mock_get_compound.assert_called_once_with("962")
        self.assertEqual(result, "water")

    @patch("iqc.pubchemtools.get_iupac_name")
    def test_get_iupac_names(self, mock_get_iupac):
        """Test getting IUPAC names for multiple compounds."""
        # Setup mocks
        mock_get_iupac.side_effect = ["water", "ethanol"]

        # Test function
        result = get_iupac_names(["962", "702"])

        # Verify results
        self.assertEqual(mock_get_iupac.call_count, 2)
        mock_get_iupac.assert_any_call("962")
        mock_get_iupac.assert_any_call("702")
        self.assertEqual(result, ["water", "ethanol"])

    @patch("iqc.pubchemtools.get_compounds")
    def test_get_cid_from_inchi(self, mock_get_compounds):
        """Test getting a CID from InChI."""
        # Setup mock
        mock_compound = MagicMock()
        mock_compound.cid = 962
        mock_get_compounds.return_value = [mock_compound]

        # Test function
        inchi = "InChI=1S/H2O/h1H2"
        result = get_cid_from_inchi(inchi)

        # Verify results
        mock_get_compounds.assert_called_once_with(inchi, "inchi")
        self.assertEqual(result, "962")

    @patch("iqc.pubchemtools.get_compounds")
    def test_get_cids_from_inchi(self, mock_get_compounds):
        """Test getting all CIDs from InChI."""
        # Setup mock with multiple compounds
        mock_compound1 = MagicMock()
        mock_compound1.cid = 962
        mock_compound2 = MagicMock()
        mock_compound2.cid = 963
        mock_get_compounds.return_value = [mock_compound1, mock_compound2]

        # Test function
        inchi = "InChI=1S/H2O/h1H2"
        result = get_cids_from_inchi(inchi)

        # Verify results
        mock_get_compounds.assert_called_once_with(inchi, "inchi")
        self.assertEqual(result, ["962", "963"])

    @patch("iqc.pubchemtools.get_compounds")
    def test_get_cid_from_smiles(self, mock_get_compounds):
        """Test getting a CID from SMILES."""
        # Setup mock
        mock_compound = MagicMock()
        mock_compound.cid = 962
        mock_get_compounds.return_value = [mock_compound]

        # Test function
        smiles = "O"
        result = get_cid_from_smiles(smiles)

        # Verify results
        mock_get_compounds.assert_called_once_with(smiles, "smiles")
        self.assertEqual(result, "962")

    @patch("iqc.pubchemtools.get_compounds")
    def test_get_cids_from_smiles(self, mock_get_compounds):
        """Test getting all CIDs from SMILES."""
        # Setup mock with multiple compounds
        mock_compound1 = MagicMock()
        mock_compound1.cid = 962
        mock_compound2 = MagicMock()
        mock_compound2.cid = 963
        mock_get_compounds.return_value = [mock_compound1, mock_compound2]

        # Test function
        smiles = "O"
        result = get_cids_from_smiles(smiles)

        # Verify results
        mock_get_compounds.assert_called_once_with(smiles, "smiles")
        self.assertEqual(result, ["962", "963"])

    @patch("iqc.pubchemtools.get_compounds")
    def test_empty_results(self, mock_get_compounds):
        """Test all functions with empty results."""
        # Setup mock for empty results
        mock_get_compounds.return_value = []

        # Test functions
        self.assertEqual(get_cid_from_inchi("invalid_inchi"), "")
        self.assertEqual(get_cids_from_inchi("invalid_inchi"), [])
        self.assertEqual(get_cid_from_smiles("invalid_smiles"), "")
        self.assertEqual(get_cids_from_smiles("invalid_smiles"), [])


if __name__ == "__main__":
    unittest.main()


class TestFailureHandling(unittest.TestCase):
    """Missing names return falsy values and failures are not cached."""

    def setUp(self):
        pubchemtools._COMPOUND_CACHE.clear()
        pubchemtools._CID_FROM_SMILES_CACHE.clear()

    @patch("iqc.pubchemtools.get_compound_from_cid")
    def test_get_iupac_name_missing_is_empty_string(self, mock_get_compound):
        """A truthy sentinel like 'Not available' defeated report.py's
        `if not iupac_name` fallback loop and leaked into reports."""
        mock_compound = MagicMock()
        mock_compound.iupac_name = None
        mock_get_compound.return_value = mock_compound

        self.assertEqual(get_iupac_name("962"), "")

        mock_get_compound.return_value = None
        self.assertEqual(get_iupac_name("962"), "")

    @patch("iqc.pubchemtools.Compound.from_cid")
    def test_transient_compound_failure_is_not_cached(self, mock_from_cid):
        """One network hiccup must not poison the CID for the process life."""
        compound = MagicMock()
        mock_from_cid.side_effect = [Exception("timeout"), compound]

        self.assertIsNone(pubchemtools.get_compound_from_cid("962"))
        self.assertIs(pubchemtools.get_compound_from_cid("962"), compound)
        self.assertEqual(mock_from_cid.call_count, 2)

    @patch("iqc.pubchemtools.get_compounds")
    def test_transient_cid_lookup_failure_is_not_cached(self, mock_get_compounds):
        good = MagicMock()
        good.cid = 962
        mock_get_compounds.side_effect = [Exception("timeout"), [good]]

        self.assertEqual(pubchemtools.get_cid_from_smiles("O"), "")
        self.assertEqual(pubchemtools.get_cid_from_smiles("O"), "962")
        self.assertEqual(mock_get_compounds.call_count, 2)
