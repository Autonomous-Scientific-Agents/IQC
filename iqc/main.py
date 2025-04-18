import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from ase.io import read

from iqc.asetools import (
    run_optimization,
    run_single_point,
    run_thermo,
    run_vibrations,
    xyz2atoms,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IQC: ASE-based quantum chemistry calculations"
    )
    parser.add_argument(
        "input",
        help="Input XYZ file or SMILES string",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--task",
        choices=["single", "opt", "vib", "thermo"],
        default="thermo",
        help="Calculation task to perform (default: thermo)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: results.json)",
        default="results.json",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fmax",
        help="Maximum force for geometry optimization (default: 0.01)",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--ignore-imag",
        help="Ignore imaginary modes in thermochemistry",
        action="store_true",
    )
    return parser.parse_args()


def save_results(results, output_file):
    """Save results to file with fallback options."""
    try:
        # First attempt: Save as JSON
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_file} in JSON format")
    except (TypeError, ValueError) as e:
        logging.warning(f"JSON serialization failed: {e}. Trying pickle...")
        try:
            # Second attempt: Save as pickle
            pickle_file = output_file.replace(".json", ".pkl")
            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)
            logging.info(f"Results saved to {pickle_file} in pickle format")
        except Exception as e:
            logging.warning(f"Pickle serialization failed: {e}. Saving as text...")
            # Third attempt: Save as text
            txt_file = output_file.replace(".json", ".txt")
            with open(txt_file, "w") as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n")
            logging.info(f"Results saved to {txt_file} in text format")


def main():
    """Main function."""
    args = parse_args()

    # Read input
    if os.path.isfile(args.input):
        atoms = xyz2atoms(args.input)
    else:
        # Assume input is SMILES string
        from iqc.asetools import get_rdmol_from_smiles, ase2rdkit2

        rdmol = get_rdmol_from_smiles(args.input, optimize=True)
        atoms = ase2rdkit2(rdmol)

    # Run calculation based on task
    if args.task == "single":
        atoms, results = run_single_point(atoms)
    elif args.task == "opt":
        atoms, results = run_optimization(atoms, fmax=args.fmax)
    elif args.task == "vib":
        atoms, results = run_vibrations(atoms)
    else:  # thermo
        atoms, results = run_thermo(
            atoms, fmax=args.fmax, ignore_imag_modes=args.ignore_imag
        )

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
