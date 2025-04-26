import json
import logging
import os
import time
from datetime import datetime
import numpy as np
from ase import Atoms, build
from ase.calculators.emt import EMT
from ase.io import read, write
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

# Optional dependencies with informative messages
XTB = None
try:
    from xtb.ase.calculator import XTB
except ImportError:
    logging.warning(
        "XTB calculator not available. Install with 'pip install xtb' if you need quantum chemistry calculations with XTB."
    )


def get_calculator(name="mace", **kwargs):
    """Initializes and returns the specified ASE calculator.

    Args:
        name (str): The name of the calculator ('mace', 'xtb', 'emt').
        **kwargs: Additional keyword arguments passed to the calculator constructor.

    Returns:
        ase.calculators.calculator.Calculator: The initialized calculator instance.
                                                 Returns EMT as a fallback if the requested
                                                 calculator is not available or fails to initialize.
    """
    name = name.lower()
    calculator = None

    if name == "mace":
        try:
            from mace.calculators import mace_mp

            mace_kwargs = {
                "model": "medium",
                "dispersion": True,
                "default_dtype": "float64",
                "device": "cpu",
                **kwargs,
            }
            try:
                # Attempt with specified/default dispersion
                calculator = mace_mp(**mace_kwargs)
                logging.info(f"Using MACE calculator with arguments: {mace_kwargs}")
            except Exception as e:
                # Try without dispersion if the first attempt failed
                logging.warning(
                    f"Failed to initialize MACE with dispersion={mace_kwargs.get('dispersion')}: {str(e)}. Trying with dispersion=False."
                )
                mace_kwargs["dispersion"] = False
                calculator = mace_mp(**mace_kwargs)
                logging.info(f"Using MACE calculator with arguments: {mace_kwargs}")
        except ImportError:
            logging.warning(
                "MACE not found. Install with 'pip install mace'. Falling back to EMT."
            )
        except RuntimeError as e:
            logging.warning(f"MACE initialization failed: {e}. Falling back to EMT.")

    elif name == "xtb":
        try:
            from xtb.ase.calculator import XTB

            xtb_kwargs = {"method": "GFN2-xTB", **kwargs}
            calculator = XTB(**xtb_kwargs)
            logging.info(f"Using XTB calculator with arguments: {xtb_kwargs}")
        except ImportError:
            logging.warning(
                "XTB not found. Install with 'pip install xtb-python'. Falling back to EMT."
            )
        except Exception as e:
            logging.warning(f"XTB initialization failed: {e}. Falling back to EMT.")

    elif name == "emt":
        try:
            from ase.calculators.emt import EMT

            calculator = EMT(**kwargs)
            logging.info(f"Using EMT calculator with arguments: {kwargs}")
        except ImportError:
            # This case should ideally not happen if ASE is installed correctly
            logging.warning(
                "EMT not found, but it's usually built-in with ASE. Problem with ASE install? Returning None."
            )
            return None

    else:
        logging.warning(f"Unknown calculator '{name}'. Falling back to EMT.")

    # Fallback to EMT if the requested calculator failed or was unknown
    if calculator is None:
        logging.warning(
            f"Calculator '{name}' failed or not found. Attempting fallback to EMT."
        )
        try:
            from ase.calculators.emt import EMT

            calculator = EMT()
            logging.info("Using EMT calculator as fallback.")
        except ImportError:
            logging.error(
                "Fallback EMT calculator could not be imported. No calculator available."
            )
            raise RuntimeError(
                "No suitable ASE calculator found or could be initialized."
            )

    return calculator


def save_atoms(atoms, prefix="", suffix="", file_format="xyz", directory=None):
    """
    Save an ASE Atoms object to a file with a name composed of:
    [prefix]_[chemical_formula]_[timestamp]_[suffix].[file_format]

    Args:
        atoms (ase.Atoms): The ASE Atoms object to be saved.
        prefix (str): Optional prefix for the filename.
        suffix (str): Optional suffix for the filename.
        file_format (str): The format in which to save the file (default is "xyz").
        directory (str): Optional directory where the file will be saved. If not provided, the file will be saved in the current directory.

    Returns:
        str: The path to the saved file.
    """
    # Get chemical formula in a canonical form
    formula = atoms.get_chemical_formula(mode="hill")

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the filename
    base_name_parts = [part for part in [prefix, formula, timestamp, suffix] if part]
    base_name = "_".join(base_name_parts) + "." + file_format

    if directory:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, base_name)
    else:
        file_path = base_name

    # Write the atoms to file
    write(file_path, atoms, format=file_format)

    return file_path


def print_atoms_info(atoms):
    """
    Prints the index, symbol, and coordinates of each atom in an ASE Atoms object.

    Args:
        atoms (ase.Atoms): The ASE Atoms object containing the atoms.

    Returns:
        None
    """
    print(f"{'Index':<6}{'Symbol':<8}{'Coordinates':<30}")
    print("-" * 44)
    for i, atom in enumerate(atoms):
        print(
            f"{i:<6}{atom.symbol:<8}{atom.position[0]:<10.3f}{atom.position[1]:<10.3f}{atom.position[2]:<10.3f}"
        )


def translate_atoms(atoms, indices, reference_index, target_index, distance):
    """
    Translates specified atoms in the direction of a vector defined by two reference atoms.

    Args:
        atoms (ase.Atoms): The ASE Atoms object containing the atoms.
        indices (list): List of indices of the atoms to be translated.
        reference_index (int): Index of the atom defining the origin of the direction vector.
        target_index (int): Index of the atom defining the target of the direction vector.
        distance (float): Distance to translate the atoms in the specified direction (in angstroms).

    Returns:
        ase.Atoms: A new ASE Atoms object with the translated atoms.
    """
    # Calculate the direction vector from reference_index to target_index
    direction_vector = atoms[target_index].position - atoms[reference_index].position
    # Normalize the direction vector
    direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
    # Compute the translation vector
    translation_vector = direction_unit_vector * distance

    # Translate the specified atoms
    for idx in indices:
        atoms[idx].position += translation_vector

    return atoms


def get_rdmol_from_smiles(smiles: str, optimize=False, seed=0xF00D):
    """Convert a SMILES string to an RDKit molecule.

    Args:
        smiles (str): The SMILES string representing the molecule.
        optimize (bool, optional): Whether to optimize the molecule geometry using MMFF. Defaults to False.
        seed (int, optional): The random seed for molecule embedding. Defaults to 0xF00D.

    Returns:
        rdkit.Chem.rdchem.Mol: The RDKit molecule object.
    """
    rdmol = AllChem.MolFromSmiles(smiles)
    rdmol = AllChem.AddHs(rdmol)
    AllChem.EmbedMolecule(rdmol, randomSeed=seed)
    if optimize:
        AllChem.MMFFOptimizeMolecule(rdmol)
    return rdmol


def get_rdmol_from_inchi(inchi: str, optimize=False, seed=0xF00D):
    """
    Convert an InChI string to an RDKit molecule.

    Args:
        inchi (str): The InChI string representing the molecule.
        optimize (bool, optional): Whether to optimize the molecule geometry using MMFF. Defaults to False.
        seed (int, optional): The random seed for molecule embedding. Defaults to 0xF00D.

    Returns:
        rdkit.Chem.rdchem.Mol: The RDKit molecule object.
    """
    rdmol = AllChem.MolFromInchi(inchi)
    rdmol = AllChem.AddHs(rdmol)
    AllChem.EmbedMolecule(rdmol, randomSeed=seed)
    if optimize:
        AllChem.MMFFOptimizeMolecule(rdmol)
    return rdmol


def get_rdmol_from_xyz(xyz: str):
    """Return RDKit molecule from xyz formatted string or a path to an xyz file

    Parameters
    ----------
    xyz : str
        xyz formatted string or a path to an xyz file

    Returns
    -------
    rdkit.Chem.rdchem.Mol
    """
    import rdkit
    from os.path import isfile

    if isfile(xyz):
        return rdkit.Chem.rdmolfiles.MolFromXYZFile(xyz)
    return rdkit.Chem.AllChem.MolFromXYZBlock(xyz)


def get_xyz_from_rdmol(rdmol):
    """
    Convert an RDKit molecule to an XYZ formatted string.

    Args:
        rdmol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.

    Returns:
        str: The XYZ formatted string representing the molecule.
    """
    conf = rdmol.GetConformer()
    xyz = str(rdmol.GetNumAtoms()) + "\n\n"
    for i in range(rdmol.GetNumAtoms()):
        at = rdmol.GetAtomWithIdx(i)
        symbol = str(at.GetSymbol())
        pos = conf.GetAtomPosition(i)
        x, y, z = pos.x, pos.y, pos.z
        xyz += f"{symbol} {x} {y} {z}\n"
    return xyz


def convert_extended_xyz_to_standard(file_path):
    """
    Reads an extended XYZ format file and converts it to standard XYZ format.

    Args:
        file_path (str): Path to the extended XYZ file.

    Returns:
        str: Standard XYZ format string.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract the number of atoms and comment line (first two lines of XYZ format)
    num_atoms = int(lines[0].strip())
    comment_line = lines[1].strip()

    # Extract the atom data lines
    atom_data = lines[2 : num_atoms + 2]

    # Convert to standard XYZ format
    standard_xyz_lines = [
        f"{line.split()[0]} {line.split()[1]} {line.split()[2]} {line.split()[3]}"
        for line in atom_data
    ]

    # Combine into the standard XYZ format string
    standard_xyz_string = f"{num_atoms}\n{comment_line}\n" + "\n".join(
        standard_xyz_lines
    )

    return standard_xyz_string


def get_canonical_smiles(molecule):
    """
    Generates the canonical SMILES string for an RDKit molecule object.

    Args:
        molecule (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        str: The canonical SMILES string, or None if the molecule is invalid.
    """
    if molecule is None:
        return None
    try:
        # Use RDKit's MolToSmiles to generate canonical SMILES
        return Chem.MolToSmiles(molecule, canonical=True)
    except Exception as e:
        print(f"Error generating SMILES: {e}")
        return None


def get_bonding_info(molecule):
    """
    Extracts bonding information for a given RDKit molecule object.

    Args:
        molecule (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        list of dict: A list of dictionaries where each dictionary represents a bond with:
            - 'atom1': Index of the first atom in the bond
            - 'atom2': Index of the second atom in the bond
            - 'atom1_symbol': Symbol of the first atom
            - 'atom2_symbol': Symbol of the second atom
            - 'bond_type': Type of the bond (e.g., SINGLE, DOUBLE)
    """
    if molecule is None:
        return []

    bonding_info = []
    for bond in molecule.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bonding_info.append(
            {
                "atom1": atom1_idx,
                "atom2": atom2_idx,
                "atom1_symbol": molecule.GetAtomWithIdx(atom1_idx).GetSymbol(),
                "atom2_symbol": molecule.GetAtomWithIdx(atom2_idx).GetSymbol(),
                "bond_type": bond_type.name,  # Get bond type as a string
            }
        )

    return bonding_info


def ase2rdkit2(atoms):
    """
    Converts an ASE Atoms object to an RDKit Mol object using RDKit's MolFromXYZBlock.

    Parameters:
        atoms (ase.Atoms): ASE Atoms object.

    Returns:
        rdkit.Chem.Mol: RDKit molecule object, or None if conversion fails.
    """
    try:
        # Export ASE object to XYZ format and read with RDKit
        xyz = f"{len(atoms)}\n\n"
        for atom, position in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            xyz += f"{atom} {position[0]} {position[1]} {position[2]}\n"

        raw_mol = Chem.MolFromXYZBlock(xyz)
        if raw_mol is None:
            raise ValueError("Failed to convert XYZ to RDKit molecule.")

        mol = Chem.Mol(raw_mol)

        # Set initial charges to 0 if missing
        if not atoms.has("initial_charges"):
            atoms.set_initial_charges([0] * len(atoms))

        # Determine bonds using RDKit
        Chem.rdDetermineBonds.DetermineBonds(
            mol, charge=int(sum(atoms.get_initial_charges()))
        )
        return mol
    except Exception as e:
        print(f"Error in ase2rdkit: {e}")
        return None


def ase2rdkit_manual(atoms, bond_threshold=1.2):
    """
    Converts an ASE Atoms object to an RDKit Mol object, manually determining bonds.

    Args:
        atoms (ase.Atoms): ASE Atoms object.
        bond_threshold (float): Bonding threshold (multiplied by covalent radii).

    Returns:
        rdkit.Chem.Mol: RDKit molecule object.
    """
    try:
        # Create an editable molecule
        mol = Chem.RWMol()

        # Add atoms to the molecule
        atomic_numbers = atoms.get_atomic_numbers()
        for atomic_num in atomic_numbers:
            mol.AddAtom(Chem.Atom(int(atomic_num)))  # Convert to Python int

        # Get positions and determine bonds
        positions = atoms.get_positions()
        radii = np.array(
            [Chem.GetPeriodicTable().GetRcovalent(int(z)) for z in atomic_numbers]
        )

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i >= j:
                    continue  # Avoid double-counting bonds

                distance = np.linalg.norm(pos_i - pos_j)
                max_bond_distance = bond_threshold * (radii[i] + radii[j])

                if distance <= max_bond_distance:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        # Convert to RDKit Mol object
        mol = mol.GetMol()
        rdmolops.SanitizeMol(mol)  # Sanitize the molecule

        return mol
    except Exception as e:
        print(f"Error in ase2rdkit_manual: {e}")
        return None


def ase2rdkit_with_bond_orders(atoms, bond_threshold=1.2):
    """
    Converts an ASE Atoms object to an RDKit Mol object, with higher-order bond detection.

    Args:
        atoms (ase.Atoms): ASE Atoms object.
        bond_threshold (float): Base bonding threshold (multiplied by covalent radii).

    Returns:
        rdkit.Chem.Mol: RDKit molecule object.
    """
    try:
        # Create an editable molecule
        mol = Chem.RWMol()

        # Add atoms to the molecule
        atomic_numbers = atoms.get_atomic_numbers()
        for atomic_num in atomic_numbers:
            mol.AddAtom(Chem.Atom(int(atomic_num)))  # Convert to Python int

        # Get positions and determine bonds
        positions = atoms.get_positions()
        radii = np.array(
            [Chem.GetPeriodicTable().GetRcovalent(int(z)) for z in atomic_numbers]
        )

        # Define bond distance thresholds for different bond types
        bond_multipliers = {
            Chem.BondType.SINGLE: 1.2,
            Chem.BondType.DOUBLE: 1.1,
            Chem.BondType.TRIPLE: 1.0,
        }

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i >= j:
                    continue  # Avoid double-counting bonds

                distance = np.linalg.norm(pos_i - pos_j)
                bond_added = False

                # Check bond type based on thresholds
                for bond_type, multiplier in bond_multipliers.items():
                    max_bond_distance = multiplier * (radii[i] + radii[j])
                    if distance <= max_bond_distance:
                        mol.AddBond(i, j, bond_type)
                        bond_added = True
                        break  # Add the highest-order bond that fits

                if not bond_added and distance <= bond_threshold * (
                    radii[i] + radii[j]
                ):
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        # Convert to RDKit Mol object
        mol = mol.GetMol()
        rdmolops.SanitizeMol(mol)  # Sanitize the molecule

        return mol
    except Exception as e:
        print(f"Error in ase2rdkit_with_bond_orders: {e}")
        return None


def get_canonical_smiles_from_atoms(atoms):
    """Convert ASE Atoms object to canonical SMILES string.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        str: Canonical SMILES string, or None if conversion fails
    """
    try:
        # Convert atoms to RDKit mol using manual bond detection
        rdmol = ase2rdkit_manual(atoms)

        if rdmol is None:
            return None

        # Generate canonical SMILES
        return Chem.MolToSmiles(rdmol, canonical=True)

    except Exception as e:
        print(f"Error converting atoms to SMILES: {e}")
        return None


atoms2smiles = get_canonical_smiles_from_atoms
rdmol2smiles = get_canonical_smiles


def ase_atoms_to_tuple(atoms):
    """
    Convert ASE Atoms object to a tuple of atom symbols and coordinates.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        tuple: A tuple containing tuples of atom symbols and their coordinates
    """
    atom_data = []
    for i in range(len(atoms)):
        atom_symbol = atoms.get_chemical_symbols()[i]
        coordinates = tuple(atoms.get_positions()[i])
        atom_data.append((atom_symbol, coordinates))
    return tuple(atom_data)


atoms2tuple = ase_atoms_to_tuple


def get_external_symmetry_factor(atoms):
    """
    Calculate the external symmetry factor for an ASE Atoms object.
    If automol is not available, returns a default value of 1.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        int: External symmetry factor
    """
    try:
        import automol

        geo = atoms2tuple(atoms)
        return automol.geom.external_symmetry_factor(geo)
    except ImportError:
        logging.warning("automol module not found. Using default symmetry factor of 1.")
        return 1
    except Exception as e:
        logging.warning(
            f"Error calculating symmetry factor: {e}. Using default value of 1."
        )
        return 1


def ase_to_rdkit_mol(atoms):
    """
    Convert ASE Atoms to an RDKit Mol.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        rdkit.Chem.Mol: RDKit molecule object
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    # Create an RDKit molecule with no bonds first.
    mol = Chem.RWMol()
    atom_indices = []
    for sym in symbols:
        a = Chem.Atom(sym)
        idx = mol.AddAtom(a)
        atom_indices.append(idx)

    # Add bonds heuristically based on distance criteria
    # This is naive and may need improvement:
    # For a real system, you'd implement a proper bond-guessing function.
    conf = Chem.Conformer(len(symbols))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, tuple(pos))
    mol.AddConformer(conf)

    # Simple bond guess: if distance < some cutoff, add a bond
    # You would refine these rules depending on your chemistry.
    distance_matrix = atoms.get_all_distances(mic=False)
    # Rough covalent radius guess table for common elements:
    cov_radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "Cl": 1.02,
        # Add more if needed
    }
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            ri = cov_radii.get(symbols[i], 0.7)
            rj = cov_radii.get(symbols[j], 0.7)
            cutoff = ri + rj + 0.4  # some margin
            if distance_matrix[i, j] < cutoff:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    rd_mol = mol.GetMol()
    # Try to sanitize molecule
    Chem.SanitizeMol(rd_mol)
    # Generate 3D coords if needed (usually we already have them from ASE)
    # But we can just keep as is, since we have coordinates from ASE.
    return rd_mol


def atoms2xyz(atoms):
    """
    Converts an ASE Atoms object to an XYZ string.

    Args:
        atoms (ase.Atoms): The ASE Atoms object to be converted.

    Returns:
        str: A string in XYZ format.
    """
    try:
        # Get the number of atoms
        num_atoms = len(atoms)

        # Create the header (number of atoms and a blank/comment line)
        xyz_str = f"{num_atoms}\n\n"

        # Add atom positions and symbols
        for symbol, position in zip(
            atoms.get_chemical_symbols(), atoms.get_positions()
        ):
            xyz_str += (
                f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}\n"
            )

        return xyz_str
    except Exception as e:
        print(f"Error in atoms_to_xyz: {e}")
        return ""


def decode_complex(dct):
    """
    Decode a complex number from a dictionary.

    Args:
        dct (dict): A dictionary containing "real" and "imag" keys.

    Returns:
        complex: The decoded complex number.
    """
    if "real" in dct and "imag" in dct:
        return complex(dct["real"], dct["imag"])
    return dct


def get_total_electrons(atoms):
    """
    Get the total number of electrons from an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        int: Total number of electrons
    """
    atomic_numbers = atoms.get_atomic_numbers()

    # Sum up all electrons
    total_electrons = sum(atomic_numbers)

    return total_electrons


def get_spin(atoms):
    """Calculate spin for an ASE Atoms object based on total electron count.

    For even number of electrons, spin=0 (singlet)
    For odd number of electrons, spin=0.5 (doublet)

    Args:
        atoms: ASE Atoms object

    Returns:
        float: Spin value (0.0 or 0.5)
    """
    total_electrons = get_total_electrons(atoms)
    return 0.5 if total_electrons % 2 else 0.0


def get_inchikey(atoms):
    """Convert ASE Atoms object to InChIKey using RDKit.

    Args:
        atoms (ase.Atoms): ASE Atoms object to convert

    Returns:
        str: InChIKey string, or empty string if conversion fails
    """
    try:
        # Convert to RDKit mol using existing function
        mol = ase2rdkit_manual(atoms)
        if mol is None:
            return ""

        # Generate InChIKey
        inchikey = Chem.MolToInchiKey(mol)
        return inchikey

    except Exception as e:
        print(f"Error in atoms2inchikey: {e}")
        return ""


atoms2inchikey = get_inchikey


def _prepare_calculation(atoms, calculator=None, unique_name=""):
    """
    Prepare atoms and calculator for a calculation.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator): The calculator instance to use.
                                                            If None, get_calculator() is called.
        unique_name (str): Unique name for the molecule

    Returns:
        tuple: (calculator, initial_data, results_dict)
    """
    if unique_name == "":
        unique_name = get_inchikey(atoms)

    # Get calculator if not provided
    if calculator is None:
        logging.debug("No calculator provided, getting default.")
        calculator = get_calculator()

    if calculator is None:
        # This should not happen if get_calculator raises RuntimeError correctly
        raise ValueError("Failed to obtain a valid calculator.")

    calc = calculator

    # Get initial data
    initial_smiles = atoms2smiles(atoms)
    initial_xyz = atoms2xyz(atoms)
    try:
        initial_sym_number = get_external_symmetry_factor(atoms)
    except Exception as e:
        logging.warning(
            f"Error getting symmetry number: {e}. Using default value of 1."
        )
        initial_sym_number = 1

    # Set calculator and get initial energy
    atoms.calc = calc
    try:
        initial_energy = atoms.get_potential_energy()
    except Exception as e:
        logging.error(f"Failed to get initial potential energy with {calc.name}: {e}")
        raise

    # Prepare results dictionary
    results = {
        "number_of_atoms": len(atoms),
        "number_of_electrons": get_total_electrons(atoms),
        "spin": get_spin(atoms),
        "formula": atoms.get_chemical_formula(mode="hill"),
        "unique_name": unique_name,
        "initial_smiles": initial_smiles,
        "initial_xyz": initial_xyz,
        "initial_sym_number": initial_sym_number,
        "initial_energy_eV": initial_energy,
        "error": None,
        "calculator_name": calc.name,
    }

    return calc, results


def run_single_point(
    atoms,
    calculator=None,
    unique_name="",
):
    """
    Run a single point energy calculation for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator, optional): Calculator instance. Defaults to None (uses get_calculator).
        unique_name (str): Unique name for the molecule

    Returns:
        tuple: A tuple containing the atoms and a dictionary with calculated properties
    """
    logging.info(f"Starting single point calculation for {unique_name}")

    calc, results = _prepare_calculation(atoms, calculator, unique_name)
    error = None

    try:
        start_time = time.time()
        # Energy is already calculated in _prepare_calculation
        energy = results["initial_energy_eV"]
        forces = atoms.get_forces()
        results["calc_time"] = (time.time() - start_time) * 1000
        results["energy_eV"] = energy
        results["forces"] = forces.tolist()
        logging.debug(
            f"Single point calculation completed in {results['calc_time']} ms"
        )
    except Exception as e:
        error = f"Error in single point calculation: {e}"
        results["error"] = error
        logging.error(error)

    logging.info(f"Single point calculation for {unique_name} completed")
    return atoms, results


def run_vibrations(
    atoms,
    calculator=None,
    unique_name="",
    indices=None,
    delta=0.01,
):
    """
    Run vibrational frequency calculations for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator, optional): Calculator instance. Defaults to None (uses get_calculator).
        unique_name (str): Unique name for the molecule
        indices (list): List of atom indices to include in vibration calculation
        delta (float): Displacement for finite difference calculation

    Returns:
        tuple: A tuple containing the atoms and a dictionary with calculated properties
    """
    calc, results = _prepare_calculation(atoms, calculator, unique_name)
    logging.info(f"Starting vibrational analysis for {unique_name} with {calc.name}")
    error = None

    try:
        start_time = time.time()
        vib = Vibrations(atoms, name=f"vib_{unique_name}", indices=indices, delta=delta)
        vib.run()
        vib_data = vib.get_vibrations()  # Get the VibrationsData object
        results["vib_time"] = (time.time() - start_time) * 1000

        # Get frequencies and energies from vib_data
        frequencies = vib_data.get_frequencies()  # cm^-1
        vib_energies = vib_data.get_energies()  # eV

        results["frequencies_cm^-1"] = (
            frequencies.tolist() if hasattr(frequencies, "tolist") else frequencies
        )
        results["vib_energies"] = (
            vib_energies.tolist() if hasattr(vib_energies, "tolist") else vib_energies
        )  # Store energies for thermo

        # We don't store the raw modes as they are complex and large
        results["modes"] = "Modes not saved (complex data)"

        # Calculate number of imaginary frequencies
        results["number_of_imaginary"] = len(
            [
                f
                for f in results["frequencies_cm^-1"]
                if isinstance(f, complex) or (isinstance(f, (int, float)) and f < 0)
            ]
        )

        logging.debug(f"Vibrational analysis completed in {results['vib_time']} ms")
    except AttributeError as ae:
        # Catch specific errors related to missing methods
        error = f"Error accessing vibration data (possibly ASE version issue?): {ae}"
        results["error"] = error
        logging.error(error)
    except Exception as e:
        error = f"Error in vibrational analysis: {e}"
        results["error"] = error
        logging.error(error)

    logging.info(f"Vibrational analysis for {unique_name} completed")
    return atoms, results


def run_optimization(
    atoms,
    calculator=None,
    fmax=0.001,
    unique_name="",
    max_steps=500,
):
    """
    Run geometry optimization for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator, optional): Calculator instance. Defaults to None (uses get_calculator).
        fmax (float): Maximum force for geometry optimization
        unique_name (str): Unique name for the molecule
        max_steps (int): Maximum number of optimization steps
        trajectory (str): Path to save trajectory file
        logfile (str): Path to save log file

    Returns:
        tuple: A tuple containing the optimized atoms and a dictionary with calculated properties
    """
    calc, results = _prepare_calculation(atoms, calculator, unique_name)
    logging.info(f"Starting geometry optimization for {unique_name} with {calc.name}")
    # Log optimization parameters
    logging.debug(f"Optimization parameters: fmax={fmax}, max_steps={max_steps}")
    error = None

    # Add optimization-specific fields
    results.update(
        {
            "opt_smiles": "",
            "opt_xyz": "",
            "opt_sym_number": 0,
            "opt_energy_eV": 0,
            "smiles_changed": None,
            "opt_time": 0,
            "opt_steps": 0,
            "opt_converged": False,
            "opt_forces": [],
        }
    )

    try:
        # Ensure atoms positions and cell are float64 before optimization
        # This can help prevent type mismatches within the optimizer
        atoms.positions = atoms.positions.astype(np.float64)
        if atoms.pbc.any():  # Check if periodic boundary conditions are set
            atoms.cell = atoms.cell.astype(np.float64)

        start_time = time.time()
        dyn = BFGS(atoms)
        converged = dyn.run(fmax=fmax, steps=max_steps)
        results["opt_time"] = (time.time() - start_time) * 1000
        results["opt_steps"] = dyn.get_number_of_steps()
        results["opt_converged"] = converged
        results["opt_forces"] = atoms.get_forces().tolist()
        logging.debug(f"Optimization completed in {results['opt_time']} ms")
    except Exception as e:
        error = f"Error in optimization: {e}"
        # Add more context to the error log
        logging.error(
            f"Optimization failed for {unique_name} using {calc.name}. Error: {e}"
        )
        if "did not contain a loop with signature matching types" in str(e):
            logging.error(
                "Potential type mismatch detected. Check calculator dtype and input geometry precision."
            )
        results["error"] = error
        # No need to log again here, already logged above

    if error is None:
        results["opt_smiles"] = atoms2smiles(atoms)
        results["opt_energy_eV"] = atoms.get_potential_energy()
        results["opt_xyz"] = atoms2xyz(atoms)
        results["opt_sym_number"] = get_external_symmetry_factor(atoms)
        results["smiles_changed"] = results["initial_smiles"] != results["opt_smiles"]

    logging.info(f"Geometry optimization for {unique_name} completed")
    return atoms, results


def run_thermo(
    atoms,
    calculator=None,
    fmax=0.01,  # Keep fmax for direct call, but will be overwritten by opt_params
    ignore_imag_modes=True,
    unique_name="",
    **opt_params,  # Accept optimization parameters
):
    """
    Run thermochemistry calculations for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator, optional): Calculator instance. Defaults to None (uses get_calculator).
        fmax (float): Maximum force for geometry optimization (can be overridden by opt_params).
        ignore_imag_modes (bool): Whether to ignore imaginary vibrational modes
        unique_name (str): Unique name for the molecule
        **opt_params: Additional keyword arguments passed to run_optimization.

    Returns:
        tuple: A tuple containing the thermochemistry results and a dictionary with calculated properties
    """

    # Get calculator first (will use default if None)
    calc = get_calculator() if calculator is None else calculator
    logging.info(
        f"Starting thermochemistry calculation for {unique_name} with {calc.name}"
    )

    # Prepare optimization arguments, merging defaults, fmax, and **opt_params
    # **opt_params will override fmax if 'fmax' is present in it
    current_opt_params = {"fmax": fmax}  # Start with default/passed fmax
    current_opt_params.update(opt_params)  # Update with params from file

    # First optimize the geometry using the obtained calculator and combined parameters
    atoms, opt_results = run_optimization(
        atoms,
        calculator=calc,
        unique_name=unique_name,
        **current_opt_params,  # Pass combined optimization params
    )
    if opt_results["error"] is not None:
        logging.error("Optimization failed, cannot proceed with thermochemistry.")
        return None, opt_results

    # Then run vibrational analysis using the same calculator
    # If vib params were added, they could be passed here similarly
    atoms, vib_results = run_vibrations(atoms, calculator=calc, unique_name=unique_name)
    if vib_results["error"] is not None:
        logging.error(
            "Vibrational analysis failed, cannot proceed with thermochemistry."
        )
        # Combine results to show both opt and vib info, even if vib failed
        results = {**opt_results, **vib_results}
        return None, results

    # Combine results
    results = {**opt_results, **vib_results}
    error = None

    try:
        start_time = time.time()
        # Get energies directly from vib_results dictionary
        vib_energies = vib_results.get("vib_energies", None)
        if vib_energies is None:
            # This case indicates an issue in run_vibrations not storing energies
            logging.error(
                "Vibrational energies not found in vibration results. Cannot calculate thermo properties."
            )
            results["error"] = "Missing vibrational energies for thermochemistry."
            return None, results

        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            geometry="nonlinear",
            atoms=atoms,
            potentialenergy=atoms.get_potential_energy(),
            spin=get_spin(atoms),
            symmetrynumber=results.get("opt_sym_number", 1),  # Use optimized symmetry
            ignore_imag_modes=ignore_imag_modes,
        )
        results["thermo_time"] = (time.time() - start_time) * 1000
        results["G_eV"] = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.0)
        results["H_eV"] = thermo.get_enthalpy(temperature=298.15)
        results["S_eV/K"] = thermo.get_entropy(temperature=298.15, pressure=101325.0)
        results["E_ZPE_eV"] = thermo.get_ZPE_correction()
        logging.debug(
            f"Thermochemistry calculations completed in {results['thermo_time']} ms"
        )
    except Exception as e:
        error = f"Error in thermochemistry calculations: {e}"
        results["error"] = error
        logging.error(error)

    logging.info(f"Thermochemistry calculation for {unique_name} completed")
    return thermo, results


def get_atoms_from_xyz(xyz, parallel=False):
    """
    Generate ASE Atoms object from XYZ input.

    Args:
        xyz (str): Either path to an XYZ file or XYZ content as string

    Returns:
        ase.Atoms: ASE Atoms object
    """
    logging.debug(f"Reading atoms from XYZ input: {xyz}")
    # Check if input is a file path
    if os.path.isfile(xyz):
        atoms = read(xyz, format="xyz", parallel=parallel)
    else:
        # Assume input is XYZ string content
        # Create temporary file to use ASE's read functionality
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz") as tmp:
            tmp.write(xyz)
            tmp.flush()
            atoms = read(tmp.name, format="xyz", parallel=parallel)

    logging.debug(f"Successfully read atoms from {xyz}")
    return atoms


xyz2atoms = get_atoms_from_xyz
