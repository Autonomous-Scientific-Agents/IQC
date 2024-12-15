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
        type=int,
        default=-1,
        help="Verbosity level of logging, 0: errors, 1: 0 + warnings, 2: 1 + info, 3: 2 + debug",
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
        "--xyzdir",
        type=str,
        default="xyz",
        help="Path for the xyz directory, where .xyz files will be read",
    )

    return parser.parse_args()
