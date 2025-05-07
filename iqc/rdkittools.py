"""
Tools for working with RDKit.
"""

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import Descriptors


def xyz_to_mol(
    xyz_text: str,
    charge: int = 0,
    determine_bonds: bool = True,
    sanitize: bool = True,
    detect_problems: bool = True,
) -> Chem.Mol:
    """Convert an XYZ text to a RDKit molecule. RDKit ≥2023.03.1 supports Chem.MolFromXYZBlock.
    Parameters
    ----------
    xyz_text : str
        The XYZ text to convert.
    charge : int, optional
        The charge of the molecule, by default 0.
    determine_bonds : bool, optional
        Whether to determine the bonds of the molecule, by default True.
    sanitize : bool, optional
        Whether to sanitize the molecule, by default True.
    detect_problems : bool, optional
        Whether to detect problems with the molecule, by default True.
    Returns
    -------
    Chem.Mol
    """
    mol = None

    # Validate input
    if not isinstance(xyz_text, str) or not xyz_text.strip():
        print("Error: XYZ text must be a non-empty string")
        return None

    try:
        # Normalize line endings to avoid platform-specific issues
        xyz_text = xyz_text.replace("\r\n", "\n").replace("\r", "\n")

        # Ensure the XYZ text is properly formatted
        lines = xyz_text.strip().split("\n")
        if len(lines) < 2:
            print("Error: XYZ text must have at least 2 lines")
            return None

        try:
            # First line should be a number (atom count)
            atom_count = int(lines[0].strip())
            if atom_count <= 0:
                print(f"Error: Invalid atom count: {atom_count}")
                return None

            # Verify we have the expected number of lines
            # format: <atom count>\n<comment>\n<atom_1>\n...<atom_n>
            if len(lines) < atom_count + 2:
                print(
                    f"Error: XYZ text has {len(lines)} lines, expected at least {atom_count + 2}"
                )
                return None
        except ValueError:
            print(f"Error: First line should be a valid integer, got: {lines[0]}")
            return None

        # Create the molecule from the XYZ block
        mol = Chem.MolFromXYZBlock(xyz_text)

    except Exception as e:
        print(f"Error converting XYZ to RDKit molecule: {e}")
        return None

    if mol is None:
        print("Error: RDKit failed to create molecule from XYZ block")
        return None

    if determine_bonds:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except Exception as e:
            print(f"Error determining bonds: {e}")
            # Continue even if bond determination fails

    if detect_problems:
        try:
            problems = Chem.DetectChemistryProblems(mol)
            if problems:
                print(f"Problems with the molecule: {problems}")
        except Exception as e:
            print(f"Error detecting chemistry problems: {e}")

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Error sanitizing RDKit molecule: {e}")

    return mol


def check_mol(mol: Chem.Mol) -> bool:
    """Check if a molecule is valid.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to check.
    Returns
    -------
    bool
    """
    Chem.SanitizeMol(mol)
    problems = Chem.DetectChemistryProblems(mol)
    if problems:
        print(f"Problems with the molecule: {problems}")
    return problems


def get_formula(mol: Chem.Mol) -> str:
    """Get the formula of a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the formula of.
    Returns
    -------
    str
    """
    return Chem.rdMolDescriptors.CalcMolFormula(mol)


def get_inchi(mol: Chem.Mol) -> str:
    """Get the InChI of a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the InChI of.
    Returns
    -------
    str
    """
    return Chem.MolToInchi(mol)


def get_smiles(
    mol: Chem.Mol,
    remove_hydrogens: bool = True,
    kekulize: bool = True,
    isomeric: bool = True,
) -> str:
    """Get the SMILES of a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the SMILES of.
    remove_hydrogens : bool, optional
        Whether to remove hydrogens from the molecule, by default True.
    kekulize : bool, optional
        Whether to kekulize the molecule, by default True.
    isomeric : bool, optional
        Whether to use isomeric SMILES, by default True.
    Returns
    -------
    str
    """
    if remove_hydrogens:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric, kekuleSmiles=kekulize)


def get_inchikey(mol: Chem.Mol) -> str:
    """Get the InChIKey of a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the InChIKey of.
    Returns
    -------
    str
    """
    return Chem.MolToInchiKey(mol)


def get_mol_weight(mol: Chem.Mol) -> float:
    """Get the molecular weight of a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the molecular weight of.
    Returns
    -------
    float
    """
    return Chem.rdMolDescriptors.CalcExactMolWt(mol)


def get_num_atoms(mol: Chem.Mol) -> int:
    """Get the number of atoms in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the number of atoms of.
    Returns
    -------
    int
    """
    return mol.GetNumAtoms()


def get_num_bonds(mol: Chem.Mol) -> int:
    """Get the number of bonds in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the number of bonds of.
    Returns
    -------
    int
    """
    return mol.GetNumBonds()


def get_num_heavy_atoms(mol: Chem.Mol) -> int:
    """Get the number of heavy atoms in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the number of heavy atoms of.
    Returns
    -------
    int
    """
    return mol.GetNumHeavyAtoms()


def get_num_hydrogens(mol: Chem.Mol) -> int:
    """Get the number of hydrogens in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the number of hydrogens of.
    Returns
    -------
    int
    """
    return Chem.rdMolDescriptors.CalcNumHBD(mol)


def get_bond_order(mol: Chem.Mol, atom1: int, atom2: int) -> int:
    """Get the bond order between two atoms.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the bond order of.
    atom1 : int
        The first atom.
    atom2 : int
        The second atom.
    Returns
    -------
    int
    """
    return mol.GetBondBetweenAtoms(atom1, atom2).GetBondTypeAsDouble()


def get_bond_orders(mol: Chem.Mol, atom: int):
    """Get the bond orders of a molecule for a given atom.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the bond orders of.
    atom : int
        The atom to get the bond orders of.
    Returns
    -------
    int
    """
    return mol.GetAtomWithIdx(atom).GetDegree()


def get_max_bond_order(mol: Chem.Mol):
    """Get the maximum bond order in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the maximum bond order of.
    Returns
    -------
    int
    """
    max_bond_order = 0
    for atom in mol.GetAtoms():
        if atom.GetDegree() > max_bond_order:
            max_bond_order = atom.GetDegree()
    return max_bond_order


def get_num_electrons(mol: Chem.Mol):
    """Get the number of electrons in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the number of electrons of.
    Returns
    -------
    int
    """
    total_electrons = 0
    for atom in mol.GetAtoms():
        total_electrons += atom.GetAtomicNum()
    total_electrons -= Chem.GetFormalCharge(mol)
    return total_electrons


def get_num_radical_electrons(mol: Chem.Mol):
    """Get the number of radical electrons in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the number of radical electrons of.
    Returns
    -------
    int
    """
    return Chem.Descriptors.NumRadicalElectrons(mol)


def get_png(mol: Chem.Mol):
    """Get the PNG image of a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the PNG image of.
    Returns
    -------
    bytes
    """
    return Chem.Draw.MolToImage(mol)
