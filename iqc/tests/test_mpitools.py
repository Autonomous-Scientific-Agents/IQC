import pytest
from unittest.mock import patch, MagicMock
import os
from iqc.mpitools import (
    SerialMPI,
    get_mpi_rank,
    get_mpi_size,
    get_mpi_local_rank,
    get_mpi_local_size,
    get_mpi_context,
    get_ppn,
    get_total_memory,
    get_start_end,
    should_initialize_mpi,
)
import sys


class MockComm:
    """A mock communicator for testing."""

    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size


# Fixtures for environment variables
@pytest.fixture
def mock_env_pmi(monkeypatch):
    monkeypatch.setenv("PMI_RANK", "1")
    monkeypatch.setenv("PMI_SIZE", "4")


@pytest.fixture
def mock_env_ompi(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")


@pytest.fixture
def clear_mpi_env(monkeypatch):
    vars_to_clear = [
        "PMI_RANK",
        "PMI_ID",
        "PMIX_RANK",
        "OMPI_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID",
        "PALS_RANKID",
        "PMI_SIZE",
        "PMIX_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_NTASKS",
        "IQC_DISABLE_MPI",
        "IQC_ENABLE_MPI",
    ]
    for var in vars_to_clear:
        monkeypatch.delenv(var, raising=False)


def test_get_mpi_rank_priority(mock_env_pmi, mock_env_ompi, monkeypatch):
    # Test that comm object has the highest priority
    mock_comm = MockComm(rank=0, size=1)
    assert get_mpi_rank(comm=mock_comm) == 0

    # Test that PMI_RANK is used when no comm is provided
    assert get_mpi_rank() == 1

    # Test that OMPI is used if PMI is not present
    monkeypatch.delenv("PMI_RANK")
    assert get_mpi_rank() == 2


def test_get_mpi_rank_no_mpi(clear_mpi_env):
    # Test that it returns 0 if MPI is not initialized and no env vars are set
    assert get_mpi_rank() == 0


def test_get_mpi_size_priority(mock_env_pmi, mock_env_ompi, monkeypatch):
    # Test that comm object has the highest priority
    mock_comm = MockComm(rank=0, size=1)
    assert get_mpi_size(comm=mock_comm) == 1

    # Test that PMI_SIZE is used when no comm is provided
    assert get_mpi_size() == 4

    # Test that OMPI is used if PMI is not present
    monkeypatch.delenv("PMI_SIZE")
    assert get_mpi_size() == 8


def test_get_mpi_size_no_mpi(clear_mpi_env):
    # Test that it returns default value if MPI is not initialized.
    assert get_mpi_size(default=2) == 2


def test_should_initialize_mpi_only_for_multi_rank_launch(clear_mpi_env):
    assert should_initialize_mpi() is False

    with patch.dict(os.environ, {"PMI_RANK": "0", "PMI_SIZE": "1"}, clear=False):
        assert should_initialize_mpi() is False

    with patch.dict(os.environ, {"PMI_RANK": "0", "PMI_SIZE": "2"}, clear=False):
        assert should_initialize_mpi() is True

    with patch.dict(
        os.environ,
        {"PMI_RANK": "0", "PMI_SIZE": "2", "IQC_DISABLE_MPI": "1"},
        clear=False,
    ):
        assert should_initialize_mpi() is False

    with patch.dict(os.environ, {"IQC_ENABLE_MPI": "1"}, clear=True):
        assert should_initialize_mpi() is True


def test_get_mpi_context_returns_serial_without_mpi_launch(clear_mpi_env):
    comm, mpi = get_mpi_context()

    assert mpi is SerialMPI
    assert comm.Get_rank() == 0
    assert comm.Get_size() == 1
    assert comm.bcast("value", root=0) == "value"
    assert comm.reduce(3, op=mpi.SUM, root=0) == 3
    assert comm.allreduce(4, op=mpi.MAX) == 4
    assert comm.gather("x", root=0) == ["x"]


def test_get_mpi_local_rank():
    with patch.dict(os.environ, {"OMPI_COMM_WORLD_LOCAL_RANK": "2"}, clear=True):
        assert get_mpi_local_rank() == 2


def test_get_mpi_local_rank_default():
    with patch.dict(os.environ, {}, clear=True):
        assert get_mpi_local_rank(default=0) == 0


def test_get_mpi_local_size():
    with patch.dict(os.environ, {"OMPI_COMM_WORLD_LOCAL_SIZE": "4"}, clear=True):
        assert get_mpi_local_size() == 4


def test_get_mpi_local_size_default():
    with patch.dict(os.environ, {}, clear=True):
        assert get_mpi_local_size(default=1) == 1


def test_get_ppn():
    with patch("os.cpu_count", return_value=8):
        assert get_ppn() == 8


def test_get_total_memory():
    class MockVirtualMemory:
        def __init__(self):
            self.total = 8 * 1024 * 1024 * 1024  # 8 GB in bytes

    with patch("psutil.virtual_memory", return_value=MockVirtualMemory()):
        assert get_total_memory() == 8192  # 8 GB in MB


class MockMPIComm:
    """Mock MPI communicator that properly inherits from MPI.Comm"""

    def __init__(self, size=4, rank=0):
        self._size = size
        self._rank = rank

    def Get_size(self):
        return self._size

    def Get_rank(self):
        return self._rank

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return self._rank


@pytest.mark.parametrize(
    "N, size, rank, expected",
    [
        (10, 4, 0, (0, 3)),  # 10 items, 4 procs, rank 0
        (10, 4, 1, (3, 6)),  # 10 items, 4 procs, rank 1
        (10, 4, 2, (6, 8)),  # 10 items, 4 procs, rank 2
        (10, 4, 3, (8, 10)),  # 10 items, 4 procs, rank 3
        (11, 4, 0, (0, 3)),  # 11 items, 4 procs, rank 0
        (11, 4, 1, (3, 6)),  # 11 items, 4 procs, rank 1
        (11, 4, 2, (6, 9)),  # 11 items, 4 procs, rank 2
        (11, 4, 3, (9, 11)),  # 11 items, 4 procs, rank 3
        (3, 5, 0, (0, 1)),  # 3 items, 5 procs, rank 0
        (3, 5, 4, (3, 3)),  # 3 items, 5 procs, rank 4
    ],
)
def test_get_start_end(N, size, rank, expected):
    """Test the workload distribution in get_start_end."""
    mock_comm = MockComm(rank=rank, size=size)
    start, end = get_start_end(mock_comm, N)
    assert (start, end) == expected
