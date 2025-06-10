from mpi4py import MPI
import os


def get_mpi_rank(comm=None):
    """
    Return mpi rank (int) if defined as an environment variable or from communicator.

    Priority:
    1. From `comm` object if provided.
    2. From environment variables (PMI_RANK, PMI_ID, OMPI_COMM_WORLD_RANK).
    3. From `MPI.COMM_WORLD` as a final fallback.

    Returns 0 if MPI is not available and no environment variables are set.
    """
    # 1. Use comm object if provided
    if comm is not None:
        try:
            # Check if it's a valid communicator
            if hasattr(comm, "Get_rank") and callable(comm.Get_rank):
                return comm.Get_rank()
        except Exception:
            # If Get_rank fails, proceed to other methods
            pass

    # 2. Check environment variables
    env_vars = ["PMI_RANK", "PMI_ID", "OMPI_COMM_WORLD_RANK"]
    for var in env_vars:
        if os.getenv(var) is not None:
            return int(os.getenv(var))

    # 3. Use global MPI as a fallback
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0  # Default rank if MPI is not available


def get_mpi_size(comm=None, default=1):
    """
    Return mpi size (int) if defined as an environment variable or from communicator.

    Priority:
    1. From `comm` object if provided.
    2. From environment variables (PMI_SIZE, OMPI_COMM_WORLD_SIZE).
    3. From `MPI.COMM_WORLD` as a final fallback.

    Returns `default` if MPI is not available and no environment variables are set.
    """
    # 1. Use comm object if provided
    if comm is not None:
        try:
            # Check if it's a valid communicator
            if hasattr(comm, "Get_size") and callable(comm.Get_size):
                return comm.Get_size()
        except Exception:
            # If Get_size fails, proceed to other methods
            pass

    # 2. Check environment variables
    env_vars = ["PMI_SIZE", "OMPI_COMM_WORLD_SIZE"]
    for var in env_vars:
        if os.getenv(var) is not None:
            return int(os.getenv(var))

    # 3. Use global MPI as a fallback
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size()
    except ImportError:
        return default  # Default size if MPI is not available


def get_mpi_local_rank(default=0):
    """
    Return mpi local rank as an integer if defined as an environment variable.

    The relative rank of this process on this node within its job.
    For example, if four processes in a job share a node, they will each be given
    a local rank ranging from 0 to 3.

    Parameters
    ----------
    default : int, optional
        Default rank to return if not in MPI environment, by default 0

    Returns
    -------
    int
        The local rank of the current process

    Notes
    -----
    See https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables
    for more information about MPI environment variables.
    """
    if os.getenv("OMPI_COMM_WORLD_LOCAL_RANK") is not None:
        rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))
    else:
        rank = default
    return rank


def get_mpi_local_size(default=1):
    """
    Return mpi local size as an integer if defined as an environment variable.

    The number of processes on this node within its job.

    Parameters
    ----------
    default : int, optional
        Default size to return if not in MPI environment, by default 1

    Returns
    -------
    int
        The number of processes on the current node

    Notes
    -----
    See https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables
    for more information about MPI environment variables.
    """
    if os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE") is not None:
        size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE"))
    else:
        size = default
    return size


def get_ppn():
    """
    Return number of processors per node.

    Returns
    -------
    int
        Number of CPUs available on the current node

    Notes
    -----
    For alternative solutions, see:
    https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
    """
    from os import cpu_count

    return cpu_count()


def get_total_memory():
    """
    Return total physical memory in MB as an integer.

    Returns
    -------
    int
        Total physical memory in megabytes (MB)

    Notes
    -----
    Requires psutil package. If not available, returns 0 and logs a debug message.
    """
    m = 0
    try:
        from psutil import virtual_memory

        mem = virtual_memory()  # In bytes
        m = mem.total >> 20  # Using bit shift to get in MB
        # m = mem.total >> 30 # Using bit shift to get in GB
    except Exception as e:
        print(f"Error getting total memory: {e}")
    return m


def get_start_end(comm, N):
    """
    Distribute N consecutive items as evenly as possible over a given communicator.

    Uneven workload (differs by at most 1) is assigned to the initial ranks.

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator
    N : int
        Total number of items to be distributed

    Returns
    -------
    start_index : int
        Index of the first local item
    end_index : int
        Index of the last local item (exclusive)

    Notes
    -----
    Indices are zero-based.
    """
    total_processes = get_mpi_size(comm)
    rank = get_mpi_rank(comm)

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
