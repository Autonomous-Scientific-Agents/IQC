#!/usr/bin/env python3
"""
pack_polymer_box.py
Generate a PACKMOL input file (and optionally run PACKMOL) to pack a polymer
into a box at a desired density.

Dependencies
------------
* Python ≥3.8
* RDKit (for molecular-weight calculation)  --  conda install -c conda-forge rdkit
* PACKMOL in your $PATH                     --  https://m3g.github.io/packmol/

Usage examples
--------------
# 50 chains, target 1.05 g/cm^3, just write input:
python pack_polymer_box.py --pdb polymer.pdb --nmol 50 --density 1.05

# Let the script decide how many chains fit in a 60 Å box at 0.9 g/cm^3:
python pack_polymer_box.py -p polymer.pdb -l 60 --density 0.9

# Generate the input and immediately run PACKMOL:
python pack_polymer_box.py -p polymer.pdb -n 75 -d 1.0 --run
"""
from pathlib import Path
import argparse
from textwrap import dedent
from rdkit import Chem
from rdkit.Chem import Descriptors
import math
import shutil
import subprocess
import sys

AVOGADRO = 6.02214076e23  #  mol⁻¹
CM_TO_ANG = 1.0e8  #  1 cm = 10⁸ Å


def molecular_weight(pdb_path: Path) -> float:
    """Return the molecular weight (g/mol) of the first MODEL in a PDB file."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    if mol is None:
        sys.exit(f"ERROR: Could not parse {pdb_path}")
    return Descriptors.ExactMolWt(mol)


def box_length_from_density(mw, n_mol, density):
    """
    Compute cubic box edge length (Å) that yields `density` (g cm⁻³)
    for n_mol molecules of molecular weight `mw` (g mol⁻¹).
    """
    mass_total_g = n_mol * mw / AVOGADRO  # g
    volume_cm3 = mass_total_g / density  # cm³
    edge_cm = volume_cm3 ** (1 / 3)  # cm
    return edge_cm * CM_TO_ANG  # Å


def nmol_from_box(mw, L, density):
    """
    Given mw (g/mol), cubic box edge L (Å) and density (g cm⁻³),
    return the integer number of molecules that best matches the density.
    """
    volume_cm3 = (L / CM_TO_ANG) ** 3  # cm³
    mass_allowed = density * volume_cm3  # g
    nmol = round(mass_allowed * AVOGADRO / mw)
    return max(nmol, 1)


def write_packmol_input(
    out_path: Path, pdb_path: Path, nmol: int, L: float, tol=2.0, seed=12345
):
    """Write a minimal PACKMOL input file."""
    template = f"""\
    tolerance {tol}
    filetype pdb
    seed {seed}

    output {out_path.with_suffix('.pdb').name}

    structure {pdb_path}
      number {nmol}
      inside box 0. 0. 0. {L:.3f} {L:.3f} {L:.3f}
    end structure
    """
    out_path.write_text(dedent(template))
    print(f"✔ Wrote PACKMOL input: {out_path}")


def run_packmol(inp: Path):
    """Run PACKMOL, feeding it the input file as text."""
    print("▶  Running PACKMOL …")
    result = subprocess.run(
        ["packmol", str(inp)],
        capture_output=True,
        text=True,  #  ← expect string I/O
    )
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(
            f"PACKMOL failed (exit {result.returncode}).\n{result.stderr}"
        )
    print("✔ PACKMOL completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and/or run a PACKMOL packing job for a polymer."
    )
    parser.add_argument(
        "-p",
        "--pdb",
        required=True,
        type=Path,
        help="Polymer PDB file (single molecule).",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-n", "--nmol", type=int, help="Number of molecules.")
    g.add_argument(
        "-l", "--length", type=float, help="Box edge length in Å. Script chooses nmol."
    )
    parser.add_argument(
        "-d",
        "--density",
        required=True,
        type=float,
        help="Target mass density (g/cm^3).",
    )
    parser.add_argument(
        "--tol", type=float, default=2.0, help="PACKMOL overlap tolerance (Å)."
    )
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for PACKMOL.")
    parser.add_argument(
        "--run", action="store_true", help="Run PACKMOL after writing the input."
    )
    parser.add_argument(
        "-o",
        "--out-prefix",
        default="packed_box",
        help="Prefix for generated PACKMOL input/output files.",
    )
    args = parser.parse_args()

    mw = molecular_weight(args.pdb)
    print(f"• Molecular weight: {mw:.3f} g/mol")

    if args.nmol:
        L = box_length_from_density(mw, args.nmol, args.density)
        nmol = args.nmol
    else:
        L = args.length
        nmol = nmol_from_box(mw, L, args.density)

    print(
        f"• Packing {nmol} molecule(s) into {L:.3f} Å cubic box "
        f"→ target density {args.density:g} g/cm³"
    )

    inp_path = Path(f"{args.out_prefix}.inp")
    write_packmol_input(inp_path, args.pdb, nmol, L, tol=args.tol, seed=args.seed)

    if args.run:
        if shutil.which("packmol") is None:
            sys.exit("ERROR: PACKMOL executable not found in PATH.")
        run_packmol(inp_path)


if __name__ == "__main__":
    main()
