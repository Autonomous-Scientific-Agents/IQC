import os


MPI = None

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_MPI_RANK_ENV_VARS = (
    "PMI_RANK",
    "PMIX_RANK",
    "PMI_ID",
    "OMPI_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
    "PALS_RANKID",
)
_MPI_SIZE_ENV_VARS = (
    "PMI_SIZE",
    "PMIX_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
    "PALS_WORLD_SIZE",
)


class SerialComm:
    """Small MPI-like communicator for single-process execution."""

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, value, root=0):
        return value

    def Barrier(self):
        return None

    def barrier(self):
        return self.Barrier()

    def reduce(self, value, op=None, root=0):
        return value

    def allreduce(self, value, op=None):
        return value

    def gather(self, value, root=0):
        return [value]

    def Abort(self, errorcode=1):
        raise SystemExit(errorcode)


class SerialMPI:
    """Minimal MPI namespace used by IQC in serial mode."""

    SUM = "sum"
    MAX = "max"
    COMM_WORLD = SerialComm()


def _env_flag(name, env=os.environ):
    value = env.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _first_int_env(names, env=os.environ):
    for name in names:
        value = env.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def should_initialize_mpi(env=os.environ):
    """Return True when the process appears to be part of a multi-rank launch."""

    if _env_flag("IQC_DISABLE_MPI", env) is True:
        return False
    if _env_flag("IQC_ENABLE_MPI", env) is True:
        return True

    rank = _first_int_env(_MPI_RANK_ENV_VARS, env)
    size = _first_int_env(_MPI_SIZE_ENV_VARS, env)
    return rank is not None and size is not None and size > 1


def get_mpi_context():
    """Return ``(comm, mpi_namespace)`` without initializing MPI for serial runs."""

    global MPI

    if not should_initialize_mpi():
        return SerialMPI.COMM_WORLD, SerialMPI

    import mpi4py

    mpi4py.rc.initialize = False
    from mpi4py import MPI as real_mpi

    if not real_mpi.Is_initialized():
        real_mpi.Init_thread(required=real_mpi.THREAD_FUNNELED)

    MPI = real_mpi
    return real_mpi.COMM_WORLD, real_mpi


def get_mpi_rank(comm=None):
    """
    Return mpi rank (int) if defined as an environment variable or from communicator.

    Priority:
    1. From `comm` object if provided.
    2. From environment variables (PMI_RANK, PMI_ID, OMPI_COMM_WORLD_RANK).
    3. From initialized `MPI.COMM_WORLD` as a final fallback.

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
    env_vars = _MPI_RANK_ENV_VARS
    for var in env_vars:
        if os.getenv(var) is not None:
            return int(os.getenv(var))

    if MPI is not None:
        return MPI.COMM_WORLD.Get_rank()

    return 0  # Default rank if MPI is not initialized


def get_mpi_size(comm=None, default=1):
    """
    Return mpi size (int) if defined as an environment variable or from communicator.

    Priority:
    1. From `comm` object if provided.
    2. From environment variables (PMI_SIZE, OMPI_COMM_WORLD_SIZE).
    3. From initialized `MPI.COMM_WORLD` as a final fallback.

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
    env_vars = _MPI_SIZE_ENV_VARS
    for var in env_vars:
        if os.getenv(var) is not None:
            return int(os.getenv(var))

    if MPI is not None:
        return MPI.COMM_WORLD.Get_size()

    return default  # Default size if MPI is not initialized


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


if __name__ == "__main__":
    """
    Test suite for mpitools.

    This block serves as a test suite for the functions in this module.
    To run this test, you need mpi4py installed and you should execute it
    with mpiexec.

    For example:
        mpiexec -n 4 python iqc/mpitools.py
    """
    comm, _mpi = get_mpi_context()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # All printing is done from rank 0 to avoid jumbled output.
    if rank == 0:
        print(f"\n--- Testing mpitools on {size} MPI processes ---")
        print("=" * 50)

    # --- Test get_mpi_rank and get_mpi_size ---
    if rank == 0:
        print("\nTesting get_mpi_rank() and get_mpi_size()...")
    # Each rank prepares its own information string
    info_str = (
        f"  Rank {rank}: get_mpi_rank() -> {get_mpi_rank()}, "
        f"get_mpi_size() -> {get_mpi_size()}"
    )
    # Gather all strings to rank 0
    all_info = comm.gather(info_str, root=0)
    if rank == 0:
        for info in all_info:
            print(info)

    info_str_comm = (
        f"  Rank {rank}: get_mpi_rank(comm) -> {get_mpi_rank(comm)}, "
        f"get_mpi_size(comm) -> {get_mpi_size(comm)}"
    )
    all_info_comm = comm.gather(info_str_comm, root=0)
    if rank == 0:
        for info in all_info_comm:
            print(info)

    # --- Test get_mpi_local_rank and get_mpi_local_size ---
    if rank == 0:
        print("\nTesting get_mpi_local_rank() and get_mpi_local_size()...")
        print("  (Note: these depend on environment variables set by the MPI launcher)")
    local_info_str = (
        f"  Rank {rank}: get_mpi_local_rank() -> {get_mpi_local_rank()}, "
        f"get_mpi_local_size() -> {get_mpi_local_size()}"
    )
    all_local_info = comm.gather(local_info_str, root=0)
    if rank == 0:
        for info in all_local_info:
            print(info)

    # --- Test get_ppn and get_total_memory (on rank 0) ---
    comm.barrier()
    if rank == 0:
        print("\nTesting get_ppn() and get_total_memory() on rank 0...")
        print(f"  get_ppn() -> {get_ppn()}")
        print(f"  get_total_memory() -> {get_total_memory()} MB")

    # --- Test get_start_end ---
    if rank == 0:
        print("\nTesting get_start_end(comm, N)...")

    # Case 1: N is a multiple of size
    N1 = 16
    if rank == 0:
        print(f"\n  Case 1: Distributing N={N1} items (evenly)")
    start1, end1 = get_start_end(comm, N1)
    workload1 = end1 - start1
    workload_info1 = f"  Rank {rank}: start={start1}, end={end1}, workload={workload1}"
    all_workload_info1 = comm.gather(workload_info1, root=0)
    if rank == 0:
        for info in all_workload_info1:
            print(info)

    # Case 2: N is not a multiple of size
    if size > 1:
        N2 = 19
        if rank == 0:
            print(f"\n  Case 2: Distributing N={N2} items (unevenly)")
        start2, end2 = get_start_end(comm, N2)
        workload2 = end2 - start2
        workload_info2 = (
            f"  Rank {rank}: start={start2}, end={end2}, workload={workload2}"
        )
        all_workload_info2 = comm.gather(workload_info2, root=0)
        if rank == 0:
            for info in all_workload_info2:
                print(info)

    if rank == 0:
        print("\n" + "=" * 50)
        print("--- mpitools testing complete ---")
