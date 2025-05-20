# coding: utf-8
"""Utility to visualize profiling results using snakeviz.

This script opens a cProfile stats file with :mod:`snakeviz`.

Example usage::

    python profiling_graph.py profile.prof
"""
import argparse
from snakeviz.cli import main as snakeviz_main


def main():
    parser = argparse.ArgumentParser(
        description="Open a profiling stats file with snakeviz"
    )
    parser.add_argument("input", help="path to a .prof stats file")
    args = parser.parse_args()

    snakeviz_main([args.input])


if __name__ == "__main__":
    main()
