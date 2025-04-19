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
from iqc.cli import get_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, "item"):  # Handle other numpy types
            return obj.item()
        return super().default(obj)

def save_results(results, output_file):
    """Save results to file with fallback options."""
    try:
        # First attempt: Save as JSON
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, cls=ComplexEncoder)
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
    args = get_args()

    # Read input
    if os.path.isfile(args.xyz):
        atoms = xyz2atoms(args.xyz)
    else:
        # Assume input is SMILES string
        from iqc.asetools import get_rdmol_from_smiles, ase2rdkit2

        rdmol = get_rdmol_from_smiles(args.xyz, optimize=True)
        atoms = ase2rdkit2(rdmol)

    # Run calculation based on task
    if args.task == "single":
        atoms, results = run_single_point(atoms)
    elif args.task == "opt":
        atoms, results = run_optimization(atoms)
    elif args.task == "vib":
        atoms, results = run_vibrations(atoms)
    else:  # thermo
        atoms, results = run_thermo(atoms)

    # Save results
    output_file = f"results_{args.task}.json"
    save_results(results, output_file)

if __name__ == "__main__":
    main()
