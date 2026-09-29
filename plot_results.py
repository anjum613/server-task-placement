"""Make the comparison figure from raw benchmark data."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


root = Path(__file__).parent
plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 12.5,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})
data = pd.read_csv(root / "results" / "raw_results.csv")
paired = data.pivot_table(index=["tasks", "case"], columns="method",
                          values="objective", aggfunc="mean")
ratios = pd.DataFrame({"GA": paired["GA"] / paired["MILP"],
                       "Greedy": paired["Greedy"] / paired["MILP"]})
summary = ratios.groupby(level="tasks").agg(["mean", "std"])
gaps = data[data.method == "MILP"].groupby("tasks").gap.agg(["mean", "std"])
sizes = np.array(sorted(data.tasks.unique()))

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
colors = {"GA": "#6B3F67", "Greedy": "#62743A"}
markers = {"GA": "D", "Greedy": "s"}
styles = {"GA": "-", "Greedy": "-."}
for method in ("GA", "Greedy"):
    mean = summary[(method, "mean")].to_numpy()
    spread = summary[(method, "std")].to_numpy()
    axes[0].errorbar(sizes, mean, yerr=spread, marker=markers[method], capsize=3,
                     linewidth=1.8, linestyle=styles[method], color=colors[method],
                     label=method)
axes[0].axhline(1, color="#3F4746", linewidth=1, linestyle="--", label="MILP incumbent")
axes[0].set(xlabel="Number of tasks", ylabel="Objective / MILP incumbent",
            xticks=sizes)
axes[0].legend(frameon=False, fontsize=11)
axes[0].grid(axis="y", alpha=0.25)
axes[0].set_title("Relative objective")

axes[1].bar(sizes.astype(str), 100 * gaps["mean"].to_numpy(),
            color="#A8643C", edgecolor="#5F4335", linewidth=0.7)
axes[1].errorbar(np.arange(len(sizes)), 100 * gaps["mean"].to_numpy(),
                 yerr=100 * gaps["std"].to_numpy(), fmt="none", color="black", capsize=3)
axes[1].set(xlabel="Number of tasks", ylabel="MILP certified gap (%)")
axes[1].grid(axis="y", alpha=0.25)
axes[1].set_axisbelow(True)
axes[1].set_title("MILP gap after one second")
fig.tight_layout()
fig.savefig(root / "results" / "comparison.png", dpi=180)
