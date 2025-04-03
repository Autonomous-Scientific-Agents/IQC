import ase
import os
import glob
import json
from datetime import datetime
from mace.calculators import mace_mp
from . import cli
from . import mpitools
from . import asetools
import logging
from mpi4py import MPI


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Get command line arguments
    args = cli.get_args()

    # Set up logging
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

    start_index, end_index = mpitools.get_start_end(comm, number_of_files)
    logging.debug(
        f"Rank {rank} processing files from index {start_index} to {end_index}."
    )

    for file in xyz_files[start_index:end_index]:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = (
            f"{os.path.splitext(os.path.basename(file))[0]}_{rank}_{time_stamp}"
        )
        logging.debug(f"Rank {rank} processing file: {file}")
        atoms = asetools.get_atoms_from_xyz(file)
        results = {
            "xyz_file": file,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mpi_size": size,
            "mpi_rank": rank,
            "hostname": os.uname().nodename,
        }
        try:
            thermo, thermo_results = asetools.run_thermo(
                atoms,
                calculators=[
                    mace_mp(
                        model="large",
                        dispersion=True,
                        default_dtype="float64",
                        device="cpu",
                    )
                ],
                fmax=0.001,
                unique_name=unique_name,
            )
            for key, val in thermo_results.items():
                results[key] = val
            logging.debug(f"Rank {rank} completed thermo calculations for file: {file}")
        except Exception as e:
            results["thermo_error"] = str(e)
            logging.error(f"Rank {rank} encountered an error: {e}")

        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            with open(f"{unique_name}_{time_stamp}.json", "w") as f:
                json.dump(results, f, indent=2, cls=asetools.ComplexEncoder)
            logging.info(f"Rank {rank} saved results for file: {file}")
        except (TypeError, ValueError) as json_error:
            logging.warning(
                f"JSON serialization failed: {json_error}. Trying pickle..."
            )
            try:
                import pickle

                with open(f"{unique_name}_{time_stamp}.pkl", "wb") as f:
                    pickle.dump(results, f)
                logging.info(f"Rank {rank} saved results using pickle for file: {file}")
            except Exception as pickle_error:
                logging.warning(
                    f"Pickle serialization failed: {pickle_error}. Trying text file..."
                )
                try:
                    with open(f"{unique_name}_{time_stamp}.txt", "w") as f:
                        for key, value in results.items():
                            f.write(f"{key}: {str(value)}\n")
                    logging.info(f"Rank {rank} saved results as text for file: {file}")
                except Exception as text_error:
                    logging.error(
                        f"All serialization attempts failed for file {file}: {text_error}"
                    )


if __name__ == "__main__":
    main()
