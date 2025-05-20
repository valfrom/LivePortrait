# coding: utf-8
"""Utility for generating graphs from profiling tables produced by ``speed.py``.

The script reads a profiling table (as printed by ``torch.profiler``) and
creates a bar chart showing the time spent in each operation.

Example usage::

    python profiling_graph.py profile.txt profile.png

"""
import argparse
import re
import matplotlib.pyplot as plt


def parse_profile_table(text):
    """Parse profiling table text into a list of ``(name, time_ms)`` tuples."""
    rows = []
    header_seen = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('Name'):
            header_seen = True
            continue
        if line.startswith('-'):
            continue
        if not header_seen:
            continue
        parts = re.split(r"\s{2,}", line)
        if not parts:
            continue
        name = parts[0]
        time_ms = None
        for part in parts[1:]:
            m = re.search(r"([0-9]*\.?[0-9]+)\s*(ms|us)", part)
            if m:
                time_ms = float(m.group(1))
                if m.group(2) == 'us':
                    time_ms /= 1000.0
                break
        if time_ms is not None:
            rows.append((name, time_ms))
    return rows


def plot_profile(results, output_path):
    """Save a horizontal bar chart from profiling results."""
    if not results:
        raise ValueError("No profiling data to plot.")
    names, times = zip(*results)
    plt.figure(figsize=(8, 0.4 * len(names) + 2))
    plt.barh(names, times, color="#348ABD")
    plt.xlabel("Time (ms)")
    plt.title("Profiling Results")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a graph image from profiling data"
    )
    parser.add_argument("input", help="text file with profiling table")
    parser.add_argument("output", help="output image path")
    parser.add_argument("--top", type=int, default=10, help="number of operations to plot")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        table = f.read()

    data = parse_profile_table(table)[: args.top]
    if not data:
        print("No data found in profiling table.")
        return

    plot_profile(data, args.output)


if __name__ == "__main__":
    main()
