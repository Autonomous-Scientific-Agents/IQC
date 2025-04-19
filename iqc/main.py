import json
import logging
import os
import pickle
import sys
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
from ase.io import read
from mpi4py import MPI

from iqc.asetools import (
    run_optimization,
    run_single_point,
    run_thermo,
    run_vibrations,
    get_atoms_from_xyz,
    get_calculator,
)
from iqc.cli import get_args
from iqc.mpitools import get_start_end

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
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get command line arguments
    args = get_args()

    # Initialize the calculator based on CLI argument
    # Do this early so any initialization errors are caught before file processing
    try:
        calculator = get_calculator(name=args.calculator)
    except RuntimeError as e:
        logging.error(f"Failed to initialize calculator: {e}")
        comm.Abort(1) # Abort MPI if calculator fails
        sys.exit(1)   # Exit if not running under MPI

    # Set up logging (after calculator init which might log warnings)
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.debug(f"Rank {rank} started with size {size}.")

    if rank == 0:
        # Check if args.xyz is a file or directory
        if os.path.isdir(args.xyz):
            xyz_dir = args.xyz
            xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
        elif os.path.isfile(args.xyz):
            xyz_files = [args.xyz]
        else:
            raise FileNotFoundError(f"Path {args.xyz} does not exist")
        number_of_files = len(xyz_files)
        logging.info(f"Rank {rank} found {number_of_files} .xyz file(s).")

    xyz_files = comm.bcast(xyz_files if rank == 0 else None, root=0)
    number_of_files = len(xyz_files)
    if number_of_files == 0:
        raise FileNotFoundError(f"No .xyz files found in directory: {args.xyz}")

    start_index, end_index = get_start_end(comm, number_of_files)
    logging.debug(
        f"Rank {rank} processing files from index {start_index} to {end_index}."
    )

    for file in xyz_files[start_index:end_index]:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = (
            f"{os.path.splitext(os.path.basename(file))[0]}_{rank}_{time_stamp}"
        )
        logging.debug(f"Rank {rank} processing file: {file}")
        
        # Read input
        try:
            if os.path.isfile(file):
                atoms = get_atoms_from_xyz(file)
            else:
                # Assume input is SMILES string
                from iqc.asetools import get_rdmol_from_smiles, ase2rdkit2
                rdmol = get_rdmol_from_smiles(file, optimize=True)
                atoms = ase2rdkit2(rdmol)
        except ValueError as e:
            logging.error(f"Rank {rank} encountered an error reading file {file}: {e}. Skipping this file.")
            continue  # Skip to the next file
        except Exception as e:
            logging.error(f"Rank {rank} encountered an unexpected error processing file {file}: {e}. Skipping this file.")
            continue # Skip to the next file

        results = {
            "xyz_file": file,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mpi_size": size,
            "mpi_rank": rank,
            "hostname": os.uname().nodename,
        }

        try:
            # Run calculation based on task using the selected calculator
            if args.task == "single":
                atoms, task_results = run_single_point(atoms, calculator)
            elif args.task == "opt":
                atoms, task_results = run_optimization(atoms, calculator)
            elif args.task == "vib":
                atoms, task_results = run_vibrations(atoms, calculator)
            else:  # thermo
                atoms, task_results = run_thermo(atoms, calculator)
            
            for key, val in task_results.items():
                results[key] = val
            logging.debug(f"Rank {rank} completed {args.task} calculations for file: {file}")
        except Exception as e:
            results[f"{args.task}_error"] = str(e)
            logging.error(f"Rank {rank} encountered an error: {e}")

        # Save results
        output_file = f"{unique_name}_{args.task}_{time_stamp}.json"
        save_results(results, output_file)

if __name__ == "__main__":
    main()
