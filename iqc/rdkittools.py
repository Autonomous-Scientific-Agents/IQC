"""
Tools for working with RDKit.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import Descriptors
import re  # Import the re module for regular expressions


def _preprocess_xyz_coordinates(xyz_text: str) -> str:
    """
    Preprocesses the XYZ coordinate lines to convert scientific notation to standard float strings.
    Handles potential errors during conversion.
    """
    lines = xyz_text.strip().split("\n")
    if len(lines) < 2:  # Must have at least atom count and comment line
        return xyz_text  # Not a valid XYZ format or too short, return as is

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        # First line is not a number, probably not a valid XYZ file
        return xyz_text

    if len(lines) < num_atoms + 2:
        # Not enough lines for the declared number of atoms
        return xyz_text

    processed_lines = lines[:2]  # Keep atom count and comment line as is

    # Regex to find numbers, including those in scientific notation
    # It matches:
    # - an optional sign (+ or -)
    # - digits
    # - an optional decimal part
    # - an optional exponent part (e or E, optional sign, digits)
    # It also handles cases where there might not be a decimal point before 'e' like "1e+00"
    scientific_notation_pattern = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")

    for i in range(2, num_atoms + 2):
        line = lines[i]
        parts = line.split()
        if not parts:  # Empty line, should not happen in coordinate section
            processed_lines.append(line)
            continue

        atom_symbol = parts[0]
        coordinates = parts[1:]

        processed_coords = []
        valid_line = True
        if len(coordinates) == 3:  # Expecting 3 coordinates
            for coord_str in coordinates:
                try:
                    # Try direct float conversion first
                    val = float(coord_str)
                    # Format to a reasonable precision, avoiding scientific notation if possible
                    # Using a general format specifier like 'g' which is often good.
                    # RDKit might prefer a fixed number of decimal places, e.g. f-string "{:.6f}"
                    # Let's try a fixed number of decimal places which is common in XYZ.
                    processed_coords.append(f"{val:.8f}")
                except ValueError:
                    # This might happen if coord_str is not a number, which would be an XYZ format error
                    # For robustness, we'll just append the original string and let RDKit handle it or fail.
                    # This also covers cases where a non-numeric string might be present by mistake.
                    processed_coords.append(coord_str)
                    valid_line = False  # Mark as potentially problematic
        else:
            # Line does not have symbol + 3 coordinates
            valid_line = False
            processed_coords = coordinates  # Keep original if format is unexpected

        if valid_line:
            processed_lines.append(f"{atom_symbol}    {'    '.join(processed_coords)}")
        else:
            # If line was problematic (e.g. wrong number of coordinates, or a coordinate couldn't be float)
            # append the original line to avoid further corruption, RDKit might still parse or fail cleanly.
            processed_lines.append(line)

    # Append any remaining lines (e.g. if the file is longer than num_atoms + 2 lines)
    processed_lines.extend(lines[num_atoms + 2 :])

    return "\n".join(processed_lines)


def xyz_to_mol(
    xyz_text: str,
    charge: int = 0,
    determine_bonds: bool = True,
    sanitize: bool = True,
    detect_problems: bool = True,
) -> Chem.Mol:
    """Convert an XYZ text to a RDKit molecule. RDKit ≥2023.03.1 supports Chem.MolFromXYZBlock.
    Pre-processes coordinates to handle scientific notation.
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
        The RDKit molecule, or None if conversion fails.
    """
    mol = None

    if not xyz_text or not isinstance(xyz_text, str):
        print("Error: XYZ input is not a valid string or is empty.")
        return None

    # Pre-process the XYZ text to handle scientific notation in coordinates
    # print(f"Original XYZ text:\n{xyz_text}") # For debugging
    processed_xyz_text = _preprocess_xyz_coordinates(xyz_text)
    # print(f"Processed XYZ text:\n{processed_xyz_text}") # For debugging

    try:
        mol = Chem.MolFromXYZBlock(processed_xyz_text)
    except Exception as e:
        # This catches errors from MolFromXYZBlock itself
        print(f"Error in RDKit's MolFromXYZBlock after preprocessing: {e}")
        print(f"Processed XYZ that caused error:\n{processed_xyz_text}")
        return None

    if mol is None:
        # MolFromXYZBlock might return None without raising an exception if parsing fails
        print("RDKit's MolFromXYZBlock returned None. Check XYZ format and content.")
        print(f"Processed XYZ that resulted in None:\n{processed_xyz_text}")
        return None

    # The rest of the operations require a valid mol object
    if determine_bonds:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except (
            RuntimeError
        ) as e:  # RDKit often raises RuntimeError for C++ level issues
            print(
                f"RuntimeError determining bonds for molecule. Charge: {charge}. Error: {e}"
            )
            # Depending on desired behavior, you might return None or the mol without bonds
            # For now, let's return None as bond determination is often critical.
            return None
        except Exception as e:
            print(f"Unexpected error determining bonds: {e}")
            return None

    if detect_problems:
        try:
            problems = Chem.DetectChemistryProblems(mol)
            if problems:
                # This is more of a warning/info, not necessarily a fatal error
                print(f"Chemistry problems detected in the molecule: {problems}")
        except Exception as e:
            print(f"Error detecting chemistry problems: {e}")
            # Continue, as this might not be fatal for all use cases

    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Error sanitizing RDKit molecule: {e}")
            # Depending on desired behavior, you might return None or the unsanitized mol
            # For now, let's return None if sanitization fails.
            return None

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

    Counts hydrogens present as explicit graph atoms (molecules built from
    XYZ) plus implicit/property hydrogens on heavy atoms (molecules built
    from SMILES without AddHs).

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the number of hydrogens of.
    Returns
    -------
    int
    """
    explicit = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
    implicit = sum(
        atom.GetTotalNumHs() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1
    )
    return explicit + implicit


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
    """Get the bond orders of the bonds involving a given atom.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the bond orders of.
    atom : int
        The atom to get the bond orders of.
    Returns
    -------
    list[float]
    """
    return [
        bond.GetBondTypeAsDouble() for bond in mol.GetAtomWithIdx(atom).GetBonds()
    ]


def get_max_bond_order(mol: Chem.Mol):
    """Get the maximum bond order in a molecule.
    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the maximum bond order of.
    Returns
    -------
    float
    """
    # Previously this returned the maximum atom degree (number of neighbors),
    # which reports e.g. 4 for methane and 3 for ethylene instead of 1 and 2.
    return max(
        (bond.GetBondTypeAsDouble() for bond in mol.GetBonds()), default=0.0
    )


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


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Get the molecule from a SMILES string.
    Parameters
    ----------
    smilest : str
        The SMILES string to get the molecule of.
    Returns
    -------
    Chem.Mol
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: could not parse SMILES string: {smiles!r}")
        return None
    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            print(f"Error: 3D embedding failed for SMILES: {smiles!r}")
            return None
        # Optimize the molecule
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        print(f"Error during 3D embedding/optimization: {e}")
        return None

    return mol


def get_mol_from_xyz_file(xyz_file: str) -> Chem.Mol:
    """Get the molecule from an XYZ file.
    Parameters
    ----------
    xyz_file : str
        The XYZ file to get the molecule of.
    Returns
    -------
    Chem.Mol
    """
    with open(xyz_file, "r") as file:
        xyz_text = file.read()
    return xyz_to_mol(xyz_text)