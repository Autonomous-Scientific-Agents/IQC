"""Tools for interacting with PubChem with pubchempy."""

import logging
from typing import Union, List, Optional, Dict, Any, Sequence
from pubchempy import get_compounds, Compound, BadRequestError

# Simple caches to avoid duplicate API calls
_COMPOUND_CACHE: Dict[str, Optional[Compound]] = {}
_CID_FROM_INCHI_CACHE: Dict[str, Any] = {}
_CID_FROM_SMILES_CACHE: Dict[str, Any] = {}


def get_compound_from_cid(cid: str) -> Optional[Compound]:
    """Get a compound from a PubChem CID.

    Parameters
    ----------
    cid : str
        The PubChem CID of the molecule.
    Returns
    -------
    Optional[Compound]
        The compound object or None if the API call failed
    """
    # Check cache first
    cache_key = str(cid)
    if cache_key in _COMPOUND_CACHE:
        return _COMPOUND_CACHE[cache_key]

    try:
        compound = Compound.from_cid(cid)
        _COMPOUND_CACHE[cache_key] = compound
        return compound
    except Exception as e:
        logging.warning(f"Error fetching compound for CID {cid}: {e}")
        _COMPOUND_CACHE[cache_key] = None
        return None


def get_compounds_from_cids(cids: Sequence[str]) -> List[Optional[Compound]]:
    """Get compounds from a list of PubChem CIDs.

    Parameters
    ----------
    cids : Sequence[str]
        A list of PubChem CIDs.
    Returns
    -------
    List[Optional[Compound]]
        A list of compound objects, with None for any CIDs that failed
    """
    return [get_compound_from_cid(cid) for cid in cids]


def get_iupac_name(cid: str) -> str:
    """Get the IUPAC name of a molecule from PubChem.

    Parameters
    ----------
    cid : str
        The PubChem CID of the molecule.

    Returns
    -------
    str
        The IUPAC name or empty string if not found.
    """
    compound = get_compound_from_cid(cid)
    if compound and hasattr(compound, "iupac_name") and compound.iupac_name:
        return compound.iupac_name
    return ""


def get_iupac_names(cids: Sequence[str]) -> List[str]:
    """Get the IUPAC names of molecules from PubChem.

    Parameters
    ----------
    cids : Sequence[str]
        List of PubChem CIDs.

    Returns
    -------
    List[str]
        A list of IUPAC names, with empty strings for any CIDs that failed.
    """
    results = []
    for cid in cids:
        results.append(get_iupac_name(cid))
    return results


def get_cid_from_inchi(inchi: str) -> str:
    """Get the PubChem CID of a molecule from an InChI string.

    Parameters
    ----------
    inchi : str
        The InChI string of the molecule.

    Returns
    -------
    str
        The CID string or an empty string if not found.
    """
    # Check cache first
    cache_key = f"{inchi}_single"
    if cache_key in _CID_FROM_INCHI_CACHE:
        return _CID_FROM_INCHI_CACHE[cache_key]

    try:
        # Use EXPLICITLY specify 'inchi' namespace
        compounds = get_compounds(inchi, "inchi")

        if not compounds:
            result = ""
        else:
            result = str(compounds[0].cid)

        # Cache the result
        _CID_FROM_INCHI_CACHE[cache_key] = result
        return result
    except Exception as e:
        logging.warning(f"Error getting CID from InChI: {e}")
        _CID_FROM_INCHI_CACHE[cache_key] = ""
        return ""


def get_cids_from_inchi(inchi: str) -> List[str]:
    """Get all PubChem CIDs of a molecule from an InChI string.

    Parameters
    ----------
    inchi : str
        The InChI string of the molecule.

    Returns
    -------
    List[str]
        List of CID strings, or an empty list if none found.
    """
    # Check cache first
    cache_key = f"{inchi}_multiple"
    if cache_key in _CID_FROM_INCHI_CACHE:
        return _CID_FROM_INCHI_CACHE[cache_key]

    try:
        # Use EXPLICITLY specify 'inchi' namespace
        compounds = get_compounds(inchi, "inchi")

        if not compounds:
            result = []
        else:
            result = [str(compound.cid) for compound in compounds]

        # Cache the result
        _CID_FROM_INCHI_CACHE[cache_key] = result
        return result
    except Exception as e:
        logging.warning(f"Error getting CIDs from InChI: {e}")
        _CID_FROM_INCHI_CACHE[cache_key] = []
        return []


def get_cid_from_smiles(smiles: str) -> str:
    """Get the PubChem CID of a molecule from a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule.

    Returns
    -------
    str
        The CID string or an empty string if not found.
    """
    # Check cache first
    cache_key = f"{smiles}_single"
    if cache_key in _CID_FROM_SMILES_CACHE:
        return _CID_FROM_SMILES_CACHE[cache_key]

    try:
        # EXPLICITLY specify 'smiles' namespace
        compounds = get_compounds(smiles, "smiles")

        if not compounds:
            result = ""
        else:
            result = str(compounds[0].cid)

        # Cache the result
        _CID_FROM_SMILES_CACHE[cache_key] = result
        return result
    except Exception as e:
        logging.warning(f"Error getting CID from SMILES: {e}")
        _CID_FROM_SMILES_CACHE[cache_key] = ""
        return ""


def get_cids_from_smiles(smiles: str) -> List[str]:
    """Get all PubChem CIDs of a molecule from a SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule.

    Returns
    -------
    List[str]
        List of CID strings, or an empty list if none found.
    """
    # Check cache first
    cache_key = f"{smiles}_multiple"
    if cache_key in _CID_FROM_SMILES_CACHE:
        return _CID_FROM_SMILES_CACHE[cache_key]

    try:
        # EXPLICITLY specify 'smiles' namespace
        compounds = get_compounds(smiles, "smiles")

        if not compounds:
            result = []
        else:
            result = [str(compound.cid) for compound in compounds]

        # Cache the result
        _CID_FROM_SMILES_CACHE[cache_key] = result
        return result
    except Exception as e:
        logging.warning(f"Error getting CIDs from SMILES: {e}")
        _CID_FROM_SMILES_CACHE[cache_key] = []
        return []


def get_pubchem_properties(cid: str) -> Optional[Dict[str, Any]]:
    """Get properties of a molecule from PubChem using Compound object.

    Parameters
    ----------
    cid : str
        The PubChem CID of the molecule.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary of compound properties or None if the API call failed
    """
    compound = get_compound_from_cid(cid)
    if not compound:
        return None

    # Convert compound attributes to dictionary
    properties = {}
    for attr in dir(compound):
        # Skip private attributes and methods
        if attr.startswith("_") or callable(getattr(compound, attr)):
            continue
        properties[attr] = getattr(compound, attr)

    return properties


def get_pubchem_properties_batch(cids: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
    """Get properties of multiple molecules from PubChem using Compound objects.

    Parameters
    ----------
    cids : Sequence[str]
        A list of PubChem CIDs.

    Returns
    -------
    List[Optional[Dict[str, Any]]]
        List of dictionaries containing compound properties, with None for any CIDs that failed
    """
    return [get_pubchem_properties(cid) for cid in cids]
