# coding: utf-8
"""Utility to visualize profiling results using gprof2dot.

This script converts a ``cProfile`` stats file to a Graphviz graph using
``gprof2dot``. ``gprof2dot`` must be installed and ``dot`` from Graphviz must be
available on ``PATH``.

Example usage::

    python profiling_graph.py profile.prof -o profile.png
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a profiling stats file to a graph using gprof2dot",
    )
    parser.add_argument("input", help="path to a .prof stats file")
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="path of the output image (e.g. profile.png)",
    )
    parser.add_argument(
        "-T",
        "--format",
        default="png",
        help="Graphviz output format (png, svg, ...)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
        proc = subprocess.Popen(
            [sys.executable, "-m", "gprof2dot", "-f", "pstats", str(input_path)],
            stdout=subprocess.PIPE,
        )
        subprocess.check_call(
            ["dot", f"-T{args.format}", "-o", str(output_path)],
            stdin=proc.stdout,
        )
        proc.stdout.close()
    else:
        subprocess.check_call(
            [sys.executable, "-m", "gprof2dot", "-f", "pstats", str(input_path)]
        )


if __name__ == "__main__":
    main()
