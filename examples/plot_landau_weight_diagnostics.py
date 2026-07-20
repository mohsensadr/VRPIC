#!/usr/bin/env python3
"""Create publication-quality plots of VRPIC weight/MxE diagnostics."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    if error.name == "matplotlib":
        sys.exit(
            "Matplotlib is required to create the figures. Install it with "
            "'python3 -m pip install matplotlib'."
        )
    raise


ALPHA_DIRECTORY = re.compile(
    r"^data_alpha_(?P<alpha>[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)$"
)

# High-contrast palette chosen to avoid yellow and brown. Keep this ordering
# fixed so a given alpha has the same color in every figure.
RUN_COLORS = ("#0072B2", "#009E73", "#CC79A7", "#D62728", "#332288")
RUN_MARKERS = ("o", "s", "^", "D", "X")
RUN_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)))
PAPER_FIGURE_SIZE = (3.5, 2.65)  # inches; suitable for a single journal column


@dataclass(frozen=True)
class Run:
    alpha: float
    steps: list[int]
    maximum_weights: list[float]
    maximum_mxe_iterations: list[int]


def read_run(directory: Path, alpha: float) -> Run:
    filename = directory / "max_weight.csv"
    required = {"step", "max_weight", "max_mxe_iterations"}
    steps: list[int] = []
    maximum_weights: list[float] = []
    maximum_mxe_iterations: list[int] = []

    with filename.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{filename} is missing columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            try:
                steps.append(int(row["step"]))
                maximum_weights.append(float(row["max_weight"]))
                maximum_mxe_iterations.append(int(row["max_mxe_iterations"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid value in {filename}:{line_number}") from error

    if not steps:
        raise ValueError(f"No diagnostic rows found in {filename}")
    return Run(alpha, steps, maximum_weights, maximum_mxe_iterations)


def discover_runs(data_root: Path) -> list[Run]:
    runs = []
    for directory in data_root.glob("data_alpha_*"):
        match = ALPHA_DIRECTORY.match(directory.name)
        if match and (directory / "max_weight.csv").is_file():
            runs.append(read_run(directory, float(match.group("alpha"))))

    if not runs:
        raise FileNotFoundError(
            f"No data_alpha_*/max_weight.csv files found below {data_root}"
        )
    return sorted(runs, key=lambda run: run.alpha)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "lines.linewidth": 1.1,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, output_stem: Path, dpi: int) -> None:
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_maximum_weights(runs: list[Run], colors: list, output: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=PAPER_FIGURE_SIZE, constrained_layout=True)
    for series_index, (run, color, marker, linestyle) in enumerate(
        zip(runs, colors, RUN_MARKERS, RUN_LINESTYLES)
    ):
        marker_offset = series_index * 1_000
        marker_indices = [
            index
            for index, step in enumerate(run.steps)
            if step >= 10_000 + marker_offset
            and (step - marker_offset) % 10_000 == 0
        ]
        axis.plot(
            run.steps,
            run.maximum_weights,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_indices,
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            label=rf"$\alpha={run.alpha:g}$",
        )

    axis.set_xlabel("Time step")
    axis.set_ylabel(r"$\|w(t)\|_\infty$")
    axis.set_yscale("log")
    axis.grid(True, which="major", color="0.88", linewidth=0.6)
    axis.grid(True, which="minor", color="0.93", linewidth=0.4)
    axis.legend(frameon=False, ncols=2, handlelength=2.2, columnspacing=0.8)
    axis.margins(x=0)
    save_figure(figure, output / "maximum_weight_vs_step", dpi)


def plot_mxe_iterations(
    runs: list[Run], colors: list, output: Path, dpi: int, scale: str
) -> None:
    figure, axis = plt.subplots(figsize=PAPER_FIGURE_SIZE, constrained_layout=True)
    for series_index, (run, color, marker, linestyle) in enumerate(
        zip(runs, colors, RUN_MARKERS, RUN_LINESTYLES)
    ):
        # Keep markers 10,000 steps apart, but stagger their phases so markers
        # from curves with identical integer values do not cover one another.
        marker_offset = series_index * 1_000
        marker_indices = [
            index
            for index, step in enumerate(run.steps)
            if step >= 10_000 + marker_offset
            and (step - marker_offset) % 10_000 == 0
        ]
        axis.plot(
            run.steps,
            run.maximum_mxe_iterations,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_indices,
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            label=rf"$\alpha={run.alpha:g}$",
        )

    axis.set_xlabel("Time step")
    axis.set_ylabel("Maximum MxE iterations")
    axis.set_yscale(scale)
    axis.grid(True, which="major", color="0.88", linewidth=0.6)
    if scale == "log":
        axis.grid(True, which="minor", color="0.93", linewidth=0.4)
    axis.legend(frameon=False, ncols=2, handlelength=2.2, columnspacing=0.8)
    axis.margins(x=0)
    save_figure(figure, output / "maximum_mxe_iterations_vs_step", dpi)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repository / "bin",
        help="directory containing data_alpha_* directories (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "bin" / "figures",
        help="figure output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--iteration-scale",
        choices=("log", "linear"),
        default="log",
        help="vertical scale for the MxE iteration plot (default: %(default)s)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    runs = discover_runs(arguments.data_root)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    if len(runs) > len(RUN_COLORS):
        raise ValueError(
            f"The fixed palette supports at most {len(RUN_COLORS)} alpha runs; "
            f"found {len(runs)}"
        )
    colors = list(RUN_COLORS[: len(runs)])

    plot_maximum_weights(runs, colors, arguments.output_dir, arguments.dpi)
    plot_mxe_iterations(
        runs,
        colors,
        arguments.output_dir,
        arguments.dpi,
        arguments.iteration_scale,
    )

    alphas = ", ".join(f"{run.alpha:g}" for run in runs)
    print(f"Plotted alpha values: {alphas}")
    print(f"Figures written to: {arguments.output_dir}")


if __name__ == "__main__":
    main()
