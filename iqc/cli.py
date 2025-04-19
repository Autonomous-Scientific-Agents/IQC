"""Command line interface for IQC.
"""

import argparse
import os


def get_args():
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
        "-s",
        "--scratch",
        type=str,
        default=os.getenv("TMPDIR", default="/tmp"),
        help="Scratch directory. If not given checks TMPDIR env. variable, if not defined uses /tmp.",
    )
    parser.add_argument(
        "-x",
        "--xyz",
        type=str,
        default="xyz",
        help="Path for an .xyz file or a directory containing .xyz files",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["single", "opt", "vib", "thermo"],
        default="thermo",
        help="Calculation task to perform (default: thermo)",
    )
    parser.add_argument(
        "-c",
        "--calculator",
        type=str,
        choices=["mace", "xtb", "emt"],
        default="mace",
        help="ASE calculator to use (default: mace)",
    )

    return parser.parse_args()
