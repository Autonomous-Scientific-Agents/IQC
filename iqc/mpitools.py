from mpi4py import MPI
import os


def get_mpi_rank(comm=None):
    """
    Return mpi rank (int) if defined as an environment variable
    """
    if comm is not None and not isinstance(comm, MPI.Comm):
        raise TypeError("comm must be an MPI communicator object")

    if os.getenv("PMI_RANK") is not None:
        rank = int(os.getenv("PMI_RANK"))
    elif os.getenv("PMI_ID") is not None:
        rank = int(os.getenv("PMI_ID"))
    elif os.getenv("OMPI_COMM_WORLD_RANK") is not None:
        rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
    else:
        rank = MPI.COMM_WORLD.Get_rank()
    return rank


def get_mpi_size(default=1):
    """
    Return mpi size (int) if defined as an environment variable
    """
    if os.getenv("PMI_SIZE") is not None:
        size = int(os.getenv("PMI_SIZE"))
    elif os.getenv("OMPI_COMM_WORLD_SIZE") is not None:
        size = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
    else:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
        except:
            size = default
    return size


def get_mpi_local_rank(default=0):
    """
    Return mpi local rank as an integer if defined as an environment variable
    https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables
    The relative rank of this process on this node within its job.
    For example, if four processes in a job share a node, they will each be given a local rank ranging from 0 to 3.
    """
    if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK") is not None:
        rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
    else:
        rank = default
    return rank


def get_mpi_local_size(default=1):
    """
    Return mpi local size as an integer if defined as an environment variable
    https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables
    The number of processes on this node within its job.
    """
    if os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE") is not None:
        size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE"))
    else:
        size = default
    return size


def get_ppn():
    """
    Return number of processors per node.
    For alternative solutions:
    https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
    """
    from os import cpu_count

    return cpu_count()


def get_total_memory():
    """
    Return totol physical memory in MB as an integer.
    """
    m = 0
    try:
        from psutil import virtual_memory

        mem = virtual_memory()  # In bytes
        m = mem.total >> 20  # Using bit shift to get in MB
        # m = mem.total >> 30 # Using bit shift to get in GB
    except:
        logging.debug(
            "psutil not found! Cannot get memory information. You can install psutil with: \n pip install psutil"
        )
    return m


def get_start_end(comm, N):
    """
    Distribute N consecutive items (rows of a matrix, blocks of a 1D array)
    as evenly as possible over a given communicator.
    Uneven workload (differs by at most 1) is assigned to the initial ranks.

    Parameters
    ----------
    comm : MPI communicator
        The MPI communicator.
    N : int
        Total number of items to be distributed.

    Returns
    -------
    start_index : int
        Index of the first local item.
    end_index : int
        Index of the last local item (exclusive).

    Notes
    -----
    Indices are zero-based.
    """
    total_processes = comm.size
    rank = comm.rank

    # Compute workload distribution
    items_per_process, remainder = divmod(N, total_processes)

    # Determine start and end indices
    if rank < remainder:
        start_index = rank * (items_per_process + 1)
        end_index = start_index + items_per_process + 1
    else:
        start_index = rank * items_per_process + remainder
        end_index = start_index + items_per_process

    return start_index, end_index
