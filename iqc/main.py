import json
import logging
import os
import pickle
import sys
import glob
from datetime import datetime
from pathlib import Path
import yaml  # Import YAML

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


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
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
    # --- Logging Setup --- (Remains mostly the same)
    logger = logging.getLogger()
    log_level_name = args.loglevel.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "IQC %(levelname)s: %(asctime)s - Rank %(mpi_rank)s - %(message)s"
    )
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(formatter)
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.mpi_rank = rank
        return record

    logging.setLogRecordFactory(record_factory)
    logging.debug(f"Number of MPI ranks: {size}.")
    # --- Load Parameters from File ---
    params = {}
    if args.params and os.path.isfile(args.params):
        try:
            with open(args.params, "r") as f:
                params = yaml.safe_load(f)
            if rank == 0:
                logging.info(f"Loaded parameters from {args.params}")
                logging.debug(f"Parameters: {params}")

        except Exception as e:
            logging.error(f"Error loading parameters from {args.params}: {e}")
            # Decide if execution should stop if params file is bad
            comm.Abort(1)
            sys.exit(1)
    elif args.params:
        if rank == 0:
            logging.warning(
                f"Parameter file specified ({args.params}) but not found. Using defaults."
            )

    # Extract specific parameter sections, defaulting to empty dicts
    calc_params = params.get("calculator_params", {})
    opt_params = params.get("optimization_params", {})
    # Example for future: thermo_params = params.get('thermo_params', {})

    # Determine calculator name: CLI > Param file > Default ('mace')
    calculator_name = args.calculator or params.get("calculator", "mace")

    # Initialize the calculator
    try:
        calculator = get_calculator(name=calculator_name, **calc_params)
    except RuntimeError as e:
        logging.error(f"Failed to initialize calculator '{calculator_name}': {e}")
        comm.Abort(1)
        sys.exit(1)

    # --- File Processing --- (Remains mostly the same)
    if rank == 0:
        if os.path.isdir(args.xyz):
            xyz_dir = args.xyz
            xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
        elif os.path.isfile(args.xyz):
            xyz_files = [args.xyz]
        else:
            # Handle non-existent path before bcast
            logging.error(
                f"Input path {args.xyz} does not exist or is not a file/directory."
            )
            xyz_files = []  # Ensure empty list is broadcast
            # Optionally abort MPI here if input is critical
            # comm.Abort(1)
            # sys.exit(1)

        number_of_files = len(xyz_files)
        if number_of_files == 0:
            logging.warning(f"No .xyz files found in {args.xyz}. Exiting.")
        else:
            logging.info(f"Found {number_of_files} .xyz file(s).")

    xyz_files = comm.bcast(xyz_files if rank == 0 else None, root=0)
    number_of_files = len(xyz_files)
    if number_of_files == 0:
        # All ranks should exit if no files
        logging.info("No files to process. Exiting.")
        sys.exit(0)

    start_index, end_index = get_start_end(comm, number_of_files)
    logging.debug(f"Processing files from index {start_index} to {end_index}.")
    # ---------------------

    for file in xyz_files[start_index:end_index]:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(file))[0]
        unique_name = f"{base_name}_{rank}_{time_stamp}"
        logging.info(f"Processing file: {file} with unique ID: {unique_name}")

        # Read input
        try:
            atoms = get_atoms_from_xyz(file)
        except ValueError as e:
            logging.error(f"Error reading file {file}: {e}. Skipping.")
            continue
        except Exception as e:
            logging.error(f"Unexpected error processing file {file}: {e}. Skipping.")
            continue

        results = {
            "xyz_file": file,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mpi_size": size,
            "mpi_rank": rank,
            "hostname": os.uname().nodename,
        }

        try:
            # Run calculation based on task using the selected calculator and parameters
            if args.task == "single":
                atoms, task_results = run_single_point(
                    atoms=atoms, calculator=calculator, unique_name=unique_name
                )
            elif args.task == "opt":
                # Pass optimization parameters from file
                atoms, task_results = run_optimization(
                    atoms=atoms,
                    calculator=calculator,
                    unique_name=unique_name,
                    **opt_params,
                )
            elif args.task == "vib":
                # Pass vibration parameters if added to config later
                # vib_params = params.get('vibration_params', {})
                atoms, task_results = run_vibrations(
                    atoms=atoms,
                    calculator=calculator,
                    unique_name=unique_name,
                    # **vib_params
                )
            else:  # thermo
                # Pass optimization and thermo parameters
                # thermo_params = params.get('thermo_params', {})
                # Decide priority for ignore_imag: CLI flag or param file?
                # Here, CLI flag takes precedence if set.
                ignore_imag = (
                    args.ignore_imag
                )  # or thermo_params.get('ignore_imag_modes', args.ignore_imag)
                atoms, task_results = run_thermo(
                    atoms=atoms,
                    calculator=calculator,
                    unique_name=unique_name,
                    ignore_imag_modes=ignore_imag,
                    **opt_params,  # Pass opt_params to run_thermo
                )

            # Merge task results into main results dict
            results.update(task_results)
            logging.debug(f"Completed {args.task} calculations for file: {file}")
        except Exception as e:
            results[f"{args.task}_error"] = str(e)
            logging.error(f"Task '{args.task}' failed for {file}: {e}", exc_info=True)

        # Save results
        output_file = f"{base_name}_{args.task}_{time_stamp}_{rank}.json"
        save_results(results, output_file)
        logging.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
