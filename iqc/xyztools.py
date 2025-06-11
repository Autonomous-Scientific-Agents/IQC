import os
from typing import (
    List,
    Tuple,
    Optional,
    Generator,
    Dict,
    Any,
    Union,
    Iterator,
)
import mmap
from dataclasses import dataclass, field
import pathlib
import pandas as pd
import io


def count_xyz_frames(
    path: Union[str, pathlib.Path], *, buffer_size: int = 4 * io.DEFAULT_BUFFER_SIZE
) -> int:
    """
    Count how many configurations (frames) are present in a multi-frame XYZ
    trajectory **without** keeping any per-frame data in memory.

    Parameters
    ----------
    path : str or pathlib.Path
        XYZ file.
    buffer_size : int, optional
        Size (bytes) of the read buffer; larger values reduce system calls
        on very large files.  Default ≈ 32 KiB × 4.

    Returns
    -------
    int
        Total number of XYZ frames.
    """
    n_frames = 0
    with open(path, "r", buffering=buffer_size, encoding="utf-8") as fh:
        while True:
            first = fh.readline()
            if not first:  # EOF
                break

            n_atoms = int(first.strip())  # first line
            fh.readline()  # second (comment) line

            # skip over atom lines (discarded immediately)
            for _ in range(n_atoms):
                fh.readline()

            n_frames += 1

    return n_frames


@dataclass
class XYZStats:
    """Container returned by `inspect_xyz`."""

    n_frames: int
    atom_counts: Optional[List[int]] = field(default=None)
    frames: Optional[List[str]] = field(default=None)

    def __iter__(self):
        """Unpack like a regular tuple if you like."""
        yield from (self.n_frames, self.atom_counts, self.frames)


# ----------------------------------------------------------------------
# Low-level generator ---------------------------------------------------
# ----------------------------------------------------------------------


def iter_xyz(path: Union[str, pathlib.Path]) -> Iterator[Tuple[int, str, List[str]]]:
    """
    Yields one configuration at a time.

    Parameters
    ----------
    path : str or Path
        Path to the XYZ file.

    Yields
    ------
    n_atoms : int
    comment : str
    atom_lines : list[str]
    """
    with open(path, "r", encoding="utf-8", buffering=1024 * 1024) as fh:
        while True:
            first = fh.readline()
            if not first:  # EOF
                break

            n_atoms = int(first.strip())  # fast int conversion
            comment = fh.readline()  # 2nd line (may be metadata)

            # Read the atom block in the tightest possible loop
            atom_lines = [fh.readline() for _ in range(n_atoms)]

            yield n_atoms, comment, atom_lines


# ----------------------------------------------------------------------


def inspect_xyz(
    path: Union[str, pathlib.Path],
    *,
    want_counts: bool = False,
    want_frames: bool = False,
) -> XYZStats:
    """
    Fast one-pass scan of an XYZ file.

    Parameters
    ----------
    path : str or Path
        Trajectory file.
    want_counts : bool, default False
        If True, also return a list with the atom count of every frame.
    want_frames : bool, default False
        If True, also return a *list of strings* where each string is a
        complete XYZ frame (number + comment + atoms).  This may be huge!

    Returns
    -------
    XYZStats
        Dataclass with fields:
        - n_frames    (always present)
        - atom_counts (None unless ``want_counts``)
        - frames      (None unless ``want_frames``)
    """
    if not want_counts and not want_frames:
        return XYZStats(count_xyz_frames(path), None, None)

    n_frames = 0
    counts: List[int] = [] if want_counts else None  # type: ignore
    frames: List[str] = [] if want_frames else None  # type: ignore

    # Tight streaming loop
    for n_atoms, comment, atom_lines in iter_xyz(path):
        n_frames += 1

        if want_counts:
            counts.append(n_atoms)

        if want_frames:
            # join *once* per frame, avoids quadratic behavior
            frames.append(f"{n_atoms}\n{comment}{''.join(atom_lines)}")

    return XYZStats(n_frames, counts, frames)


class XYZConfiguration:
    """Lightweight class to store XYZ configuration data"""

    __slots__ = ["num_atoms", "comment", "atoms"]

    def __init__(
        self,
        num_atoms: int,
        comment: str = "",
        atoms: List[Tuple[str, float, float, float]] = None,
    ):
        self.num_atoms = num_atoms
        self.comment = comment
        self.atoms = atoms or []


class XYZReader:
    """High-performance XYZ file reader optimized for large files"""

    def __init__(self, filename: str):
        self.filename = filename
        self.file_size = os.path.getsize(filename)

    def count_configurations(self) -> int:
        """Fast count of configurations without loading data into memory"""
        count = 0
        with open(self.filename, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                pos = 0
                while pos < len(mmapped_file):
                    # Find end of line for atom count
                    line_end = mmapped_file.find(b"\n", pos)
                    if line_end == -1:
                        break

                    try:
                        # Parse atom count
                        num_atoms = int(mmapped_file[pos:line_end].decode().strip())
                        pos = line_end + 1

                        # Skip comment line
                        comment_end = mmapped_file.find(b"\n", pos)
                        if comment_end == -1:
                            break
                        pos = comment_end + 1

                        # Skip atom lines
                        for _ in range(num_atoms):
                            atom_end = mmapped_file.find(b"\n", pos)
                            if atom_end == -1:
                                return count
                            pos = atom_end + 1

                        count += 1
                    except (ValueError, UnicodeDecodeError):
                        # Skip malformed lines
                        pos = line_end + 1

        return count

    def get_atom_counts(self) -> List[int]:
        """Get list of atom counts for each configuration"""
        atom_counts = []
        with open(self.filename, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                pos = 0
                while pos < len(mmapped_file):
                    line_end = mmapped_file.find(b"\n", pos)
                    if line_end == -1:
                        break

                    try:
                        num_atoms = int(mmapped_file[pos:line_end].decode().strip())
                        atom_counts.append(num_atoms)
                        pos = line_end + 1

                        # Skip comment line
                        comment_end = mmapped_file.find(b"\n", pos)
                        if comment_end == -1:
                            break
                        pos = comment_end + 1

                        # Skip atom lines
                        for _ in range(num_atoms):
                            atom_end = mmapped_file.find(b"\n", pos)
                            if atom_end == -1:
                                return atom_counts
                            pos = atom_end + 1

                    except (ValueError, UnicodeDecodeError):
                        pos = line_end + 1

        return atom_counts

    def iter_configurations(self) -> Generator[XYZConfiguration, None, None]:
        """Memory-efficient iterator over configurations"""
        with open(self.filename, "r", encoding="utf-8", buffering=8192 * 16) as f:
            while True:
                # Read atom count
                line = f.readline()
                if not line:
                    break

                try:
                    num_atoms = int(line.strip())
                except ValueError:
                    continue

                # Read comment
                comment = f.readline().strip()

                # Read atoms
                atoms = []
                for _ in range(num_atoms):
                    atom_line = f.readline()
                    if not atom_line:
                        return

                    parts = atom_line.strip().split()
                    if len(parts) >= 4:
                        try:
                            symbol = parts[0]
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            atoms.append((symbol, x, y, z))
                        except ValueError:
                            continue

                yield XYZConfiguration(num_atoms, comment, atoms)

    def read_all_configurations(self) -> List[XYZConfiguration]:
        """Read all configurations into memory (use with caution for large files)"""
        return list(self.iter_configurations())

    def get_configuration_at_index(self, index: int) -> Optional[XYZConfiguration]:
        """Get a specific configuration by index (0-based)"""
        for i, config in enumerate(self.iter_configurations()):
            if i == index:
                return config
        return None

    def analyze_file(
        self, load_configurations: bool = False, get_atom_counts: bool = False
    ) -> Dict[str, Any]:
        """Comprehensive file analysis with configurable options"""
        result = {
            "filename": self.filename,
            "file_size_bytes": self.file_size,
            "num_configurations": 0,
        }

        if get_atom_counts:
            atom_counts = self.get_atom_counts()
            result["atom_counts"] = atom_counts
            result["num_configurations"] = len(atom_counts)

            if atom_counts:
                result["min_atoms"] = min(atom_counts)
                result["max_atoms"] = max(atom_counts)
                result["avg_atoms"] = sum(atom_counts) / len(atom_counts)
        else:
            result["num_configurations"] = self.count_configurations()

        if load_configurations:
            result["configurations"] = self.read_all_configurations()

        return result


# Convenience functions
def read_xyz_file(
    filename: str,
    count_only: bool = True,
    get_atom_counts: bool = False,
    load_configurations: bool = False,
) -> Dict[str, Any]:
    """
    Main function to read XYZ files with various options

    Args:
        filename: Path to XYZ file
        count_only: If True, only count configurations (fastest)
        get_atom_counts: If True, return list of atom counts per configuration
        load_configurations: If True, load all configurations into memory

    Returns:
        Dictionary with analysis results
    """
    reader = XYZReader(filename)

    if count_only and not get_atom_counts and not load_configurations:
        return {
            "filename": filename,
            "num_configurations": reader.count_configurations(),
        }

    return reader.analyze_file(
        load_configurations=load_configurations, get_atom_counts=get_atom_counts
    )


def count_xyz_configurations(filename: str) -> int:
    """Fast function to just count configurations"""
    return XYZReader(filename).count_configurations()


def get_xyz_atom_counts(filename: str) -> List[int]:
    """Get list of atom counts for each configuration"""
    return XYZReader(filename).get_atom_counts()


def xyz_to_dataframe(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    """
    Convert XYZ file(s) to a pandas DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to either a single XYZ file or a directory containing XYZ files.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - number_of_atoms: int
        - comment: str
        - xyz_string: str (complete XYZ frame as string)
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    # Handle single file
    if path.is_file():
        reader = XYZReader(str(path))
        data = []
        for config in reader.iter_configurations():
            xyz_str = f"{config.num_atoms}\n{config.comment}"
            for atom in config.atoms:
                xyz_str += f"\n{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}"
            data.append(
                {
                    "number_of_atoms": config.num_atoms,
                    "comment": config.comment,
                    "xyz_string": xyz_str,
                }
            )
        return pd.DataFrame(data)

    # Handle directory
    elif path.is_dir():
        data = []
        for xyz_file in path.glob("*.xyz"):
            reader = XYZReader(str(xyz_file))
            for config in reader.iter_configurations():
                xyz_str = f"{config.num_atoms}\n{config.comment}"
                for atom in config.atoms:
                    xyz_str += f"\n{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}"
                data.append(
                    {
                        "number_of_atoms": config.num_atoms,
                        "comment": config.comment,
                        "xyz_string": xyz_str,
                    }
                )
        return pd.DataFrame(data)

    else:
        raise ValueError(f"Path {path} does not exist or is not a file/directory")


def dataframe_to_xyz(
    df: pd.DataFrame,
    path: Union[str, pathlib.Path],
    xyz_column: Optional[str] = None,
):
    """
    Writes an XYZ file from a pandas DataFrame.

    The DataFrame must contain a column with XYZ frame strings.
    This function searches for a column named 'xyz_string' or 'xyz' by default.
    The column name can be specified with the `xyz_column` parameter.

    Each entry in the column is treated as a single XYZ frame. This function
    ensures frames are properly separated by newlines in the output file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the XYZ data.
    path : str or pathlib.Path
        Path for the output XYZ file.
    xyz_column : str, optional
        Name of the column with XYZ frame strings. If not provided,
        the function searches for 'xyz_string' and then 'xyz'.
    """
    if xyz_column is None:
        if "xyz_string" in df.columns:
            xyz_column = "xyz_string"
        elif "xyz" in df.columns:
            xyz_column = "xyz"
        else:
            raise ValueError(
                "DataFrame must contain a column with XYZ frame data. "
                "Specify `xyz_column`, or use 'xyz_string' or 'xyz' as column name."
            )
    elif xyz_column not in df.columns:
        raise ValueError(f"Specified column '{xyz_column}' not found in DataFrame.")

    # A generator expression is memory-efficient.
    # strip() handles frames with or without trailing newlines.
    content = "\n".join(str(frame).strip() for frame in df[xyz_column])

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        if content:
            f.write("\n")
