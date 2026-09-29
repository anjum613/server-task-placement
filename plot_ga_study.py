"""Plot paired GA cost changes, retaining the variation between instance means."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "ga_tuning")
    root = parser.parse_args(argv).results
    plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 12.5,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    data = pd.read_csv(root / "raw_results.csv")
    paired = data.pivot(index=["tasks", "case", "repeat"], columns="configuration",
                        values="objective")
    changes = 100 * (paired.divide(paired["baseline"], axis=0) - 1)
    case_changes = changes.groupby(level=["tasks", "case"]).mean()
    labels = {
        "baseline": "Default settings",
        "population_12": "Population 12",
        "population_48": "Population 48",
        "tournament_2": "Tournament size 2",
        "tournament_5": "Tournament size 5",
        "rank_selection": "Rank selection",
        "crossover_0": "Crossover 0",
        "crossover_06": "Crossover 0.6",
        "mutation_0": "Mutation 0",
        "mutation_001": "Mutation 0.01",
        "mutation_008": "Mutation 0.08",
    }
    names = [name for name in labels if name in changes.columns]
    sizes = sorted(data.tasks.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(10.2, 5.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    colors = ["#77643A", "#6D506B"]  # ochre brown and muted mulberry
    markers = ["s", "o"]
    positions = np.arange(len(names))
    for index, (ax, tasks) in enumerate(zip(axes, sizes)):
        panel = case_changes.loc[tasks, names]
        shifts = np.linspace(-0.15, 0.15, len(panel))
        for shift, (_, row) in zip(shifts, panel.iterrows()):
            ax.scatter(row.to_numpy(), positions + shift, s=23, marker=markers[index % 2],
                       facecolors="none", edgecolors=colors[index % 2], alpha=0.65,
                       linewidths=0.9, zorder=3)
        ax.scatter(panel.mean().to_numpy(), positions, s=48, marker="D",
                   color=colors[index % 2], edgecolor="white", linewidth=0.45, zorder=4)
        ax.axvline(0, color="#414745", linestyle="--", linewidth=1)
        ax.set_title(f"{tasks} tasks, {int(data[data.tasks == tasks].servers.iloc[0])} servers")
        ax.set_yticks(positions, [labels[name] for name in names])
        ax.grid(axis="x", color="#B8B5AB", alpha=0.35, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Cost change from paired baseline (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="D", color="none", markerfacecolor="#555D58",
                      markeredgecolor="white", markersize=7, label="Mean across three instances"),
               Line2D([], [], marker="o", color="none", markerfacecolor="none",
                      markeredgecolor="#555D58", markersize=5, label="One instance, five paired runs")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(root / "parameter_changes.png", dpi=220)
    fig.savefig(root / "parameter_changes.pdf")
    plt.close(fig)
    print(root / "parameter_changes.png")


if __name__ == "__main__":
    main()
