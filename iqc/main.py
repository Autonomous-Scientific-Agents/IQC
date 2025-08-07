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
import ase  # just to disable parallel features of ASE, import it before mpi initialization
import ase.parallel as asepar
import time

asepar.world = asepar.DummyMPI()
from iqc.asetools import (
    run_optimization,
    run_single_point,
    run_thermo,
    run_vibrations,
    get_atoms_from_xyz,
    get_calculator,
    get_ase_version,
)
from mpi4py import MPI

from iqc.xyztools import count_xyz_frames
from iqc.cli import get_args
from iqc.mpitools import get_start_end

from iqc.databasetools import (
	create_database,
	insert_entry
)


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

def insert_jsonl_to_db(jsonl_file, db_path):
    with open(jsonl_file, "r") as f:
        for line in f:
            try:
                insert_entry(line, db_path)
                logging.info(f"Inserted entry from JSONL: {line}")
            except Exception as e:
                logging.error(f"Error inserting entry: {e}")

def main():
    """Main function."""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    start_time = time.time()

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
            sys.exit(1)
            comm.Abort(1)

    elif args.params:
        if rank == 0:
            logging.warning(
                f"Parameter file specified ({args.params}) but not found. Using defaults."
            )
    task = args.task
    # Extract specific parameter sections, defaulting to empty dicts
    calc_params = params.get("calculator_params", {})
    opt_params = params.get("optimization_params", {})
    vib_params = params.get("vibration_params", {})
    thermo_params = params.get("thermo_params", {})
    opt_params = {**calc_params, **opt_params}
    vib_params = {**opt_params, **vib_params}
    thermo_params = {**vib_params, **thermo_params}
    if rank == 0:
        logging.info(f"All parameters: {params}")
    # Determine calculator name: CLI > Param file > Default ('mace')
    calculator_name = args.calculator or params.get("calculator", "mace")

    # Initialize the calculator
    try:
        calculator = get_calculator(name=calculator_name, **calc_params)
    except RuntimeError as e:
        logging.error(f"Failed to initialize calculator '{calculator_name}': {e}")
        comm.Abort(1)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{'tmp'}_{task}_{rank}_{time_stamp}"
    logging.debug(f"Creating directory: {dir_name}")
    os.makedirs(dir_name, exist_ok=True)
    if rank == 0:
        if os.path.isdir(args.xyz):
            xyz_dir = args.xyz
            xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
            number_of_xyz = len(xyz_files)
        elif os.path.isfile(args.xyz):
            xyz_files = [args.xyz]
            try:
                number_of_xyz = count_xyz_frames(args.xyz)
            except Exception as e:
                logging.error(f"Error counting .xyz files in {args.xyz}: {e}")
                comm.Abort(1)
        else:
            # Handle non-existent path before bcast
            logging.error(
                f"Input path {args.xyz} does not exist or is not a file/directory."
            )
            xyz_files = []  # Ensure empty list is broadcast
            comm.Abort(1)

        number_of_files = len(xyz_files)
        if number_of_files == 0:
            logging.error(f"No .xyz files found in {args.xyz}. Exiting.")
            comm.Abort(1)
        else:
            logging.info(f"Found {number_of_files} .xyz file(s).")
            logging.info(f"Number of configurations: {number_of_xyz}")

    xyz_files = comm.bcast(xyz_files if rank == 0 else None, root=0)
    number_of_xyz = comm.bcast(number_of_xyz if rank == 0 else None, root=0)
    number_of_files = len(xyz_files)

    start_index, end_index = get_start_end(comm, number_of_xyz)
    logging.debug(f"Processing files from index {start_index} to {end_index}.")
    logging.info(f"Initialization time: {time.time() - start_time} seconds.")
    for xyz_index in range(start_index, end_index):
        if number_of_files > 1:
            xyz_file = xyz_files[xyz_index]
        else:  # only one file
            xyz_file = xyz_files[0]
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(xyz_file))[0]
        unique_name = f"{base_name}_{xyz_index}_{rank}_{time_stamp}"
        logging.info(f"Processing file: {xyz_file} with unique ID: {unique_name}")

        # Read input
        try:
            if number_of_files > 1:
                atoms = get_atoms_from_xyz(xyz_file)
            else:
                atoms = get_atoms_from_xyz(xyz_file, index=xyz_index)
        except ValueError as e:
            logging.error(f"Error reading file {xyz_file}: {e}. Skipping.")
            continue
        except Exception as e:
            logging.error(
                f"Unexpected error processing file {xyz_file}: {e}. Skipping."
            )
            continue

        results = {
            "xyz_file": xyz_file,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mpi_size": size,
            "mpi_rank": rank,
            "hostname": os.uname().nodename,
            "ase_version": get_ase_version(),
            "task": task,
            "calculator": calculator_name,
        }

        try:
            # Run calculation based on task using the selected calculator and parameters
            if task == "single":
                atoms, task_results = run_single_point(
                    atoms=atoms, calculator=calculator, unique_name=unique_name
                )
            elif task == "opt":
                # Pass optimization parameters from file
                trajectory_file = None
                save_geometry = False
                if args.save:
                    trajectory_file = f"{unique_name}_opt_trajectory.traj"
                    save_geometry = True
                atoms, task_results = run_optimization(
                    atoms=atoms,
                    calculator=calculator,
                    unique_name=unique_name,
                    trajectory=trajectory_file,
                    save_geometry=save_geometry,
                    **opt_params,
                )
            elif task == "vib":
                # Pass vibration parameters if added to config later
                # vib_params = params.get('vibration_params', {})
                vib_params["vib_dir"] = dir_name
                trajectory_file = None
                save_geometry = False
                if args.save:
                    trajectory_file = f"{unique_name}_vib_trajectory.traj"
                    save_geometry = True
                atoms, task_results = run_vibrations(
                    atoms=atoms,
                    calculator=calculator,
                    optimize=True,
                    unique_name=unique_name,
                    trajectory=trajectory_file,
                    save_geometry=save_geometry,
                    **vib_params,
                )
            else:  # thermo
                # Pass optimization and thermo parameters
                # thermo_params = params.get('thermo_params', {})
                # Decide priority for ignore_imag: CLI flag or param file?
                # Here, CLI flag takes precedence if set.
                ignore_imag = (
                    args.ignore_imag
                )  # or thermo_params.get('ignore_imag_modes', args.ignore_imag)
                thermo_params["vib_dir"] = dir_name
                trajectory_file = None
                save_geometry = False
                if args.save:
                    trajectory_file = f"{unique_name}_thermo_trajectory.traj"
                    save_geometry = True
                atoms, task_results = run_thermo(
                    atoms=atoms,
                    calculator=calculator,
                    unique_name=unique_name,
                    ignore_imag_modes=ignore_imag,
                    trajectory=trajectory_file,
                    save_geometry=save_geometry,
                    **thermo_params,
                )

            # Merge task results into main results dict
            results.update(task_results)
            logging.debug(f"Completed {task} calculations for file: {xyz_file}")
        except Exception as e:
            results[f"{task}_error"] = str(e)
            logging.error(f"Task '{task}' failed for {xyz_file}: {e}", exc_info=True)

        # Save results
        output_file = f"{unique_name}_{task}_{time_stamp}_{rank}.json"
        output_file = os.path.join(dir_name, output_file)
        save_results(results, output_file)

    # Wait for all processes to finish before combining files
    barrier_start = time.time()
    logging.debug(f"Waiting for all processes to finish before combining files.")
    comm.Barrier()
    logging.debug(f"Took { time.time() - barrier_start:.2f} seconds")

    if rank == 0:
        combine_start = time.time()
        logging.debug(f"Starting to combine JSON files")

        # Combine all JSON files into a single JSONL file
        jsonl_file = f"iqc_{task}_results_{time_stamp}.jsonl"
        json_files = glob.glob(os.path.join("tmp*", f"*_{task}_*.json"), recursive=True)
        logging.debug(f"Found {len(json_files)} JSON files to combine.")
        with open(jsonl_file, "w") as outfile:
            for json_file in json_files:
                with open(json_file, "r") as infile:
                    data = json.load(infile)
                    json.dump(data, outfile)
                    outfile.write("\n")

        combine_end = time.time()
        logging.debug(
            f"Finished combining JSON files in {combine_end - combine_start:.2f} seconds"
        )
        logging.info(f"Combined results saved to {jsonl_file}")
        logging.info(f"Total time: {time.time() - start_time} seconds.")

    # SQLite database

    if not os.path.exists(jsonl_file):
        logging.error(f"JSONL file not found: {jsonl_file}")
        sys.exit(1) 

    db_path = args.database

    if db_path:  

        logging.info(f"Checking/Creating database at: {db_path}")
        create_database(db_path)  

        logging.info(f"Reading from JSONL file: {jsonl_file}")
        insert_jsonl_to_db(jsonl_file, db_path)
        logging.info(f"Finished inserting data into database: {db_path}")

    else:
        logging.info("No database specified.")

    return 0


if __name__ == "__main__":
    main()
