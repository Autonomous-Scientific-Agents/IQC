import os
import tempfile
import pathlib
import pandas as pd
import pytest

from iqc import xyztools

SIMPLE_XYZ = """3
water molecule
O 0.000000 0.000000 0.000000
H 0.757000 0.586000 0.000000
H -0.757000 0.586000 0.000000
2
hydrogen molecule
H 0.000000 0.000000 0.000000
H 0.740000 0.000000 0.000000
"""

@pytest.fixture
def xyz_file(tmp_path):
    file = tmp_path / "test.xyz"
    file.write_text(SIMPLE_XYZ)
    return file

def test_count_xyz_frames(xyz_file):
    assert xyztools.count_xyz_frames(xyz_file) == 2

def test_iter_xyz(xyz_file):
    frames = list(xyztools.iter_xyz(xyz_file))
    assert len(frames) == 2
    assert frames[0][0] == 3
    assert "water molecule" in frames[0][1]
    assert frames[1][0] == 2

def test_inspect_xyz_counts(xyz_file):
    stats = xyztools.inspect_xyz(xyz_file, want_counts=True)
    assert stats.n_frames == 2
    assert stats.atom_counts == [3, 2]
    assert stats.frames is None

def test_inspect_xyz_frames(xyz_file):
    stats = xyztools.inspect_xyz(xyz_file, want_frames=True)
    assert stats.n_frames == 2
    assert stats.frames is not None
    assert stats.frames[0].startswith("3\nwater molecule")

def test_XYZReader_count(xyz_file):
    reader = xyztools.XYZReader(str(xyz_file))
    assert reader.count_configurations() == 2

def test_XYZReader_atom_counts(xyz_file):
    reader = xyztools.XYZReader(str(xyz_file))
    assert reader.get_atom_counts() == [3, 2]

def test_XYZReader_iter_configurations(xyz_file):
    reader = xyztools.XYZReader(str(xyz_file))
    configs = list(reader.iter_configurations())
    assert len(configs) == 2
    assert configs[0].num_atoms == 3
    assert configs[1].num_atoms == 2
    assert configs[0].atoms[0][0] == "O"

def test_XYZReader_get_configuration_at_index(xyz_file):
    reader = xyztools.XYZReader(str(xyz_file))
    config = reader.get_configuration_at_index(1)
    assert config is not None
    assert config.num_atoms == 2

def test_read_xyz_file_count_only(xyz_file):
    result = xyztools.read_xyz_file(str(xyz_file))
    assert result["num_configurations"] == 2

def test_read_xyz_file_atom_counts(xyz_file):
    result = xyztools.read_xyz_file(str(xyz_file), count_only=False, get_atom_counts=True)
    assert result["atom_counts"] == [3, 2]

def test_xyz_to_dataframe_and_back(xyz_file, tmp_path):
    df = xyztools.xyz_to_dataframe(xyz_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    out_file = tmp_path / "out.xyz"
    xyztools.dataframe_to_xyz(df, out_file)
    # Check round-trip
    df2 = xyztools.xyz_to_dataframe(out_file)
    assert df2.equals(df)

def test_xyz_to_dataframe_directory(tmp_path):
    # Create two xyz files
    f1 = tmp_path / "a.xyz"
    f2 = tmp_path / "b.xyz"
    f1.write_text(SIMPLE_XYZ)
    f2.write_text(SIMPLE_XYZ)
    df = xyztools.xyz_to_dataframe(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4

def test_dataframe_to_xyz_column_selection(xyz_file, tmp_path):
    df = xyztools.xyz_to_dataframe(xyz_file)
    df = df.rename(columns={"xyz_string": "xyz"})
    out_file = tmp_path / "col.xyz"
    xyztools.dataframe_to_xyz(df, out_file, xyz_column="xyz")
    assert out_file.read_text().startswith("3\nwater molecule")
