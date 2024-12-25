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
from mace.calculators import mace_mp
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from xtb.ase.calculator import XTB


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        int: External symmetry factor
    """
    import automol

    geo = atoms2tuple(atoms)
    return automol.geom.external_symmetry_factor(geo)


default_calculator = mace_mp(
    model="medium", dispersion=True, default_dtype="float32", device="cpu"
)


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


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)


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


def run_thermo(
    atoms,
    calculators=[default_calculator],
    fmax=0.01,
    ignore_imag_modes=True,
    unique_name="",
):
    """
    Run thermochemistry calculations for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculators (list): List of calculators for geometry and frequency calculations
        fmax (float): Maximum force for geometry optimization
        ignore_imag_modes (bool): Whether to ignore imaginary vibrational modes
        unique_name (str): Unique name for the molecule

    Returns:
        tuple: A tuple containing the thermochemistry results and a dictionary with calculated properties
    """
    logging.info(f"Starting thermochemistry calculation for {unique_name}")
    
    if unique_name == "":
        unique_name = get_inchikey(atoms)

    # Ensure calculators is a list; if only one is provided, use it for all steps
    if not isinstance(calculators, list):
        calculators = [calculators]
    if len(calculators) == 1:
        calc_geom = calculators[0]
        calc_freq = calculators[0]
    else:
        calc_geom = calculators[0]
        calc_freq = calculators[1]

    initial_smiles = atoms2smiles(atoms)
    initial_xyz = atoms2xyz(atoms)
    initial_sym_number = get_external_symmetry_factor(atoms)

    # Set the first calculator and calculate energy
    atoms.calc = calc_geom
    initial_energy = atoms.get_potential_energy()
    error = None

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
        "error": error,
        "opt_smiles": "",
        "opt_xyz": "",
        "opt_sym_number": 0,
        "opt_energy_eV": 0,
        "smiles_changed": None,
        "frequencies_cm^-1": [],
        "number_of_imaginary": -1,
        "G_eV": 0,
        "H_eV": 0,
        "S_eV/K": 0,
        "E_ZPE_eV": 0,
        "opt_time": 0,
        "vib_time": 0,
        "thermo_time": 0,
    }

    # Optimize geometry
    try:
        start_time = time.time()
        dyn = BFGS(atoms)
        dyn.run(fmax=fmax)
        results["opt_time"] = (time.time() - start_time) * 1000
        logging.debug(f"Optimization completed in {results['opt_time']} ms")
    except Exception as e:
        error = f"Error in optimization: {e}"
        results["error"] = error
        logging.error(error)

    # After optimization, get optimized SMILES
    if error is None:
        results["opt_smiles"] = atoms2smiles(atoms)
        results["opt_energy_eV"] = atoms.get_potential_energy()
        results["opt_xyz"] = atoms2xyz(atoms)
        results["opt_sym_number"] = get_external_symmetry_factor(atoms)
        results["smiles_changed"] = initial_smiles != results["opt_smiles"]
        # Now set the frequency calculator and compute vibrational frequencies
        atoms.calc = calc_freq
        vib = Vibrations(atoms, name=f"vib_{unique_name}", indices=None)
        try:
            start_time = time.time()
            vib.run()
            results["vib_time"] = (
                time.time() - start_time
            ) * 1000  # Convert to milliseconds
            logging.debug(f"Vibrational analysis completed in {results['vib_time']} ms")
        except Exception as e:
            error = f"Error in vibrations: {e}"
            results["error"] = error
            logging.error(error)

    if error is None:
        start_time = time.time()
        freqs = vib.get_vibrations(read_cache=False).get_frequencies()  # in cm^-1
        results["frequencies_cm^-1"] = freqs.tolist()
        thermo = IdealGasThermo(
            vib_energies=vib.get_energies(),  # in eV
            geometry="nonlinear",  # guess or determine the molecular geometry type
            atoms=atoms,
            potentialenergy=atoms.get_potential_energy(),
            spin=get_spin(atoms),  # adjust if needed
            symmetrynumber=get_external_symmetry_factor(atoms),
            ignore_imag_modes=ignore_imag_modes,
        )
        # Save standard thermochemical properties
        results["thermo_time"] = (
            time.time() - start_time
        ) * 1000  # Convert to milliseconds
        logging.debug(
            f"Thermochemistry calculations completed in {results['thermo_time']} ms"
        )
        results["number_of_imaginary"] = thermo.n_imag
        results["G_eV"] = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.0)
        results["H_eV"] = thermo.get_enthalpy(temperature=298.15)
        results["S_eV/K"] = thermo.get_entropy(temperature=298.15, pressure=101325.0)
        results["E_ZPE_eV"] = thermo.get_ZPE_correction()

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
