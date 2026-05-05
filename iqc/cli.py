"""Command line interface for IQC."""

import argparse
import os
import sys


def add_bool_flag(parser, name, default=None, help_text=""):
    """Add paired --flag / --no-flag boolean options."""

    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{dest.replace('_', '-')}",
        dest=dest,
        action="store_false",
        help=f"Disable {dest.replace('_', ' ')}.",
    )
    parser.set_defaults(**{dest: default})


def _explicit_option_names(argv):
    """Return option names explicitly provided on the command line."""

    options = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            continue
        option = token.split("=", 1)[0]
        if option.startswith("--"):
            options.add(option)
        else:
            options.add(option[:2])
    return options


def get_args(argv=None):
    """
    Returns args object that contains command line options.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
    Command line arguments for IQC
    """,
    )

    parser.add_argument(
        "-l",
        "--loglevel",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "-f",
        "--logfile",
        type=str,
        default="none",
        help="Log file prefix, use none for logging to STDOUT, include DATE if you want a date stamp",
    )
    parser.add_argument(
        "--scratch",
        type=str,
        default=os.getenv("TMPDIR", default="/tmp"),
        help="Scratch directory. If not given checks TMPDIR env. variable, if not defined uses /tmp.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help=(
            "Path to a tabular data file to inspect. Supports parquet, CSV/TSV, "
            "Excel, JSON/JSONL, Feather, and Arrow IPC formats."
        ),
    )
    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        help=(
            "Column name used to sort tabular --input rows before running a "
            "calculation."
        ),
    )
    parser.add_argument(
        "--sort-order",
        "--sort_order",
        dest="sort_order",
        choices=["up", "down"],
        default="up",
        help="Sort order for --sort: up for ascending or down for descending.",
    )
    parser.add_argument(
        "-x",
        "--xyz",
        type=str,
        default="xyz",
        help=(
            "Path for an .xyz file or a directory containing .xyz files. When "
            "--input is provided, this is the column name containing XYZ text."
        ),
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default=None,
        help=(
            "SMILES string to convert to a 3D geometry with RDKit. When --input "
            "is provided, this is the column name containing SMILES strings."
        ),
    )
    parser.add_argument(
        "--min-natom",
        type=int,
        default=None,
        help="Minimum number of atoms for filtering molecules",
    )
    parser.add_argument(
        "--max-natom",
        type=int,
        default=None,
        help="Maximum number of atoms for filtering molecules",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["single", "opt", "vib", "ir", "thermo", "nmr"],
        default="thermo",
        help="Calculation task to perform (default: thermo)",
    )
    parser.add_argument(
        "-c",
        "--calculator",
        type=str,
        choices=[
            "mace",
            "xtb",
            "emt",
            "uma",
            "uma-s-omol",
            "uma-s-omat",
            "uma-s-odac",
            "uma-m-omol",
            "uma-m-omat",
            "uma-m-odac",
        ],
        default="mace",
        help="ASE calculator to use (default: mace)",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help="Path to YAML file with calculator and optimization parameters.",
    )
    parser.add_argument(
        "--ignore-imag",
        help="Ignore imaginary modes in thermochemistry (can also be set in param file)",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default=None,
        help="Path to insert data into SQLite database",
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Save trajectory and optimized structures to file",
        action="store_true",
    )
    parser.add_argument(
        "--direct-db",
        help="Save results directly to the SQLite database without creating any files.",
        action="store_true",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["orca", "nwchem", "gaussian", "xtb"],
        default=None,
        help="Electronic-structure backend for NMR calculations. xTB is only supported as --optimization-backend.",
    )
    parser.add_argument(
        "--optimization-backend",
        type=str,
        choices=["orca", "nwchem", "gaussian", "xtb"],
        default=None,
        help="Backend for optional geometry optimization before NMR.",
    )
    parser.add_argument(
        "--nuclei",
        nargs="+",
        default=None,
        help="Nuclei to compute for NMR, e.g. --nuclei 1H 13C",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method for NMR shielding calculations.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default=None,
        help="Basis set for NMR shielding calculations.",
    )
    parser.add_argument(
        "--optimization-method",
        type=str,
        default=None,
        help="Method for optional geometry optimization before NMR.",
    )
    parser.add_argument(
        "--optimization-basis",
        type=str,
        default=None,
        help="Basis set for optional geometry optimization before NMR.",
    )
    parser.add_argument(
        "--solvent-model",
        type=str,
        default=None,
        help="Implicit solvent model for NMR calculations, e.g. smd or cpcm.",
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default=None,
        help="Solvent name for implicit-solvent NMR calculations.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=None,
        help=(
            "Total molecular charge. Used by NMR backends and by ASE "
            "calculators that support charge (e.g. xTB, FAIRChem UMA). "
            "Overrides any 'charge=' set in the XYZ comment line."
        ),
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=None,
        help=(
            "Spin multiplicity 2S+1 (singlet=1, doublet=2, triplet=3, ...). "
            "Used by NMR backends and by ASE calculators that support spin "
            "(e.g. xTB, FAIRChem UMA). Overrides any 'multiplicity='/'uhf=' "
            "in the XYZ comment line."
        ),
    )
    add_bool_flag(
        parser,
        "--optimize-geometry",
        default=None,
        help_text="Optimize the geometry before the NMR calculation.",
    )
    add_bool_flag(
        parser,
        "--conformer-sampling",
        default=None,
        help_text="Perform RDKit-based conformer sampling before NMR.",
    )
    parser.add_argument(
        "--num-conformers",
        type=int,
        default=None,
        help="Maximum number of conformers to retain for NMR calculations.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature in Kelvin for Boltzmann weighting.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=None,
        help="Lorentzian/Gaussian line broadening in ppm for NMR spectra.",
    )
    parser.add_argument(
        "--lineshape",
        type=str,
        choices=["lorentzian", "gaussian", "pseudo-voigt"],
        default=None,
        help="Line shape used to simulate the NMR spectrum.",
    )
    parser.add_argument(
        "--plot-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_PPM", "MAX_PPM"),
        help="Chemical-shift plotting range in ppm, applied to each requested nucleus.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for NMR plots and tabulated outputs.",
    )
    parser.add_argument(
        "--reference-shielding",
        action="append",
        default=None,
        help="Reference shielding override in the form nucleus=value, e.g. 1H=31.77",
    )

    args = parser.parse_args(argv)
    explicit_options = _explicit_option_names(sys.argv[1:] if argv is None else argv)
    args.input_only = bool(args.input) and explicit_options <= {"-i", "--input"}
    args.input_xyz_column = bool(args.input) and bool(
        explicit_options.intersection({"-x", "--xyz"})
    )
    args.input_smiles_column = bool(args.input) and "--smiles" in explicit_options
    args.sort_order_explicit = bool(
        explicit_options.intersection({"--sort-order", "--sort_order"})
    )
    return args
