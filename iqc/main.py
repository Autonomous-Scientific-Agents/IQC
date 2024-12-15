from mpi4py import MPI
import os
import glob
import json
from . import cli
from . import mpitools
from . import asetools


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Get command line arguments
    args = cli.get_args()

    # Only rank 0 should list the files
    if rank == 0:
        xyzdir = args.xyzdir
        # Create a list of all .xyz files in the specified directory
        xyz_files = glob.glob(os.path.join(xyzdir, "*.xyz"))
        number_of_files = len(xyz_files)
        print(f"Rank {rank} found {number_of_files} .xyz files in {xyzdir}")

    # Broadcast the list of files to all ranks if needed
    xyz_files = comm.bcast(xyz_files if rank == 0 else None, root=0)
    number_of_files = len(xyz_files)

    start_index, end_index = mpitools.get_start_end(comm, number_of_files)
    for file in xyz_files[start_index:end_index]:
        file_name = os.path.splitext(os.path.basename(file))[0]
        atoms = asetools.get_atoms_from_xyz(file)
        smiles = asetools.atoms2smiles(atoms)
        results = {
            "name": file,
            "number_of_atoms": int(len(atoms)),
            "number_of_electrons": int(asetools.get_total_electrons(atoms)),
            "spin": asetools.get_spin(atoms),
            "formula": atoms.get_chemical_formula(mode="hill"),
            "smiles": smiles,
        }
        with open(f"{file_name}.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
