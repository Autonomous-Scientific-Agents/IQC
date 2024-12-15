import pytest
from unittest.mock import patch, MagicMock
import os
from iqc.mpitools import (
    get_mpi_rank,
    get_mpi_size,
    get_mpi_local_rank,
    get_mpi_local_size,
    get_ppn,
    get_total_memory,
    get_start_end,
)


class MockMPI:
    class COMM_WORLD:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 4

        @property
        def size(self):
            return 4

        @property
        def rank(self):
            return 0


@pytest.fixture
def mock_mpi():
    """Mock MPI module"""
    with patch("iqc.mpitools.MPI", MockMPI()):
        yield


def test_get_mpi_rank_from_pmi(mock_mpi):
    with patch.dict(os.environ, {"PMI_RANK": "2"}, clear=True):
        assert get_mpi_rank() == 2


def test_get_mpi_rank_from_pmi_id(mock_mpi):
    with patch.dict(os.environ, {"PMI_ID": "3"}, clear=True):
        assert get_mpi_rank() == 3


def test_get_mpi_rank_from_ompi(mock_mpi):
    with patch.dict(os.environ, {"OMPI_COMM_WORLD_RANK": "1"}, clear=True):
        assert get_mpi_rank() == 1


def test_get_mpi_size_from_pmi(mock_mpi):
    with patch.dict(os.environ, {"PMI_SIZE": "4"}, clear=True):
        assert get_mpi_size() == 4


def test_get_mpi_size_from_ompi(mock_mpi):
    with patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "8"}, clear=True):
        assert get_mpi_size() == 8


def test_get_mpi_size_default(mock_mpi):
    with patch.dict(os.environ, {}, clear=True):
        with patch("iqc.mpitools.MPI", side_effect=ImportError):
            assert get_mpi_size(default=1) == 1


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


class MockMPIForStartEnd:
    class COMM_WORLD:
        def __init__(self, size=4, rank=0):
            self._size = size
            self._rank = rank

        @property
        def size(self):
            return self._size

        @property
        def rank(self):
            return self._rank


@pytest.mark.parametrize(
    "N,rank,expected",
    [
        (10, 0, (0, 3)),  # First process gets 3 items
        (10, 1, (3, 6)),  # Second process gets 3 items
        (11, 2, (6, 9)),  # Third process gets 3 items
        (11, 3, (9, 11)),  # Fourth process gets 2 items
    ],
)
def test_get_start_end(N, rank, expected):
    mock_comm = MockMPIForStartEnd.COMM_WORLD(size=4, rank=rank)
    with patch("iqc.mpitools.MPI.COMM_WORLD", mock_comm):
        start, end = get_start_end(mock_comm, N)
        assert (start, end) == expected
