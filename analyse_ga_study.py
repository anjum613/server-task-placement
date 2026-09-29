"""Generate the measured GA tables and select a setting using tuning data only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def configuration_labels() -> dict[str, str]:
    return {
        "baseline": "Default",
        "population_12": r"Population $N_{\rm pop}=12$",
        "population_48": r"Population $N_{\rm pop}=48$",
        "tournament_2": "Tournament $k=2$",
        "tournament_5": "Tournament $k=5$",
        "rank_selection": "Rank selection",
        "crossover_0": "Crossover $p_c=0$",
        "crossover_06": "Crossover $p_c=0.6$",
        "mutation_0": "Mutation $p_m=0$",
        "mutation_001": "Mutation $p_m=0.01$",
        "mutation_008": "Mutation $p_m=0.08$",
    }


def read_complete_study(root: Path) -> list[dict]:
    metadata = json.loads((root / "metadata.json").read_text())
    with (root / "raw_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(metadata["run_order"]):
        raise ValueError(f"Study is incomplete: {root}")
    if any(row["status"] != "feasible" for row in rows):
        raise ValueError("The report tables require all runs to have a checked feasible cost")
    return rows


def change_cell(value: float) -> str:
    return f"${value:+.2f}$" if abs(value) >= 0.005 else "$0.00$"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path, default=Path(__file__).parent / "ga_tuning")
    parser.add_argument("--validation", type=Path, default=Path(__file__).parent / "ga_validation")
    args = parser.parse_args(argv)
    tuning_rows = read_complete_study(args.tuning)
    with (args.tuning / "summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    ranking = {}
    for row in summary:
        ranking.setdefault(row["configuration"], []).append(float(row["mean_paired_change_pct"]))
    pooled = {name: float(np.mean(values)) for name, values in ranking.items()}
    selected = min(pooled, key=pooled.get)
    selection = {"configuration": selected,
                 "criterion": "minimum mean paired cost change; equal weight to each size and instance",
                 "pooled_tuning_change_pct": pooled[selected], "all_configurations": pooled}
    (args.tuning / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    labels = configuration_labels()
    table = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{GA settings at three seconds. Each size has three instances and five runs per setting. Cost and SD are in CU; SD is the average within instance sample standard deviation. Change is the mean paired percentage change from the default.}",
             r"\label{tab:ga-tuning}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{100 tasks, 12 servers} & \multicolumn{3}{c}{200 tasks, 24 servers} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Setting & Cost & Change (\%) & SD & Cost & Change (\%) & SD \\", r"\midrule"]
    for name, label in labels.items():
        if name not in ranking:
            continue
        values = []
        for tasks in (100, 200):
            row = next(r for r in summary if r["configuration"] == name and int(r["tasks"]) == tasks)
            values.extend([f"{float(row['mean_objective']):.2f}",
                           change_cell(float(row['mean_paired_change_pct'])),
                           f"{float(row['mean_within_instance_sd']):.2f}"])
        table.append(label + " & " + " & ".join(values) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (args.tuning / "summary_table.tex").write_text("\n".join(table) + "\n")

    validation = read_complete_study(args.validation)
    if {row["configuration"] for row in validation} != {"baseline", selected}:
        raise ValueError("Validation must compare the tuning-selected configuration with baseline")
    tuning_cases = {row["case"] for row in tuning_rows}
    if tuning_cases & {row["case"] for row in validation}:
        raise ValueError("Tuning and validation case sets overlap")
    validation_table = [r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Validation on unseen instance seeds at three seconds, with five runs per cell. SD is the sample standard deviation within that instance, in CU. Changes pair identical instance and run seeds.}",
        r"\label{tab:ga-validation}", r"\begin{tabular}{rrrrrrr}", r"\toprule",
        r"Tasks & Seed & Default (CU) & Selected (CU) & Change (\%) & Default SD & Selected SD \\",
        r"\midrule"]
    case_results = []
    for case in sorted({row["case"] for row in validation}):
        group = [r for r in validation if r["case"] == case]
        base = {int(r["repeat"]): float(r["objective"]) for r in group if r["configuration"] == "baseline"}
        candidate = {int(r["repeat"]): float(r["objective"]) for r in group if r["configuration"] == selected}
        if set(base) != set(candidate):
            raise ValueError("Unmatched validation repetitions")
        changes = [100 * (candidate[rep] / base[rep] - 1) for rep in sorted(base)]
        result = {"case": case, "tasks": int(group[0]["tasks"]), "seed": int(group[0]["instance_seed"]),
                  "mean_default_cost": float(np.mean(list(base.values()))),
                  "mean_selected_cost": float(np.mean(list(candidate.values()))),
                  "default_sd": float(np.std(list(base.values()), ddof=1)),
                  "selected_sd": float(np.std(list(candidate.values()), ddof=1)),
                  "mean_paired_change_pct": float(np.mean(changes)),
                  "paired_wins": sum(candidate[r] < base[r] for r in base), "repetitions": len(base)}
        case_results.append(result)
        values = [str(result["tasks"]), str(result["seed"]),
                  f"{result['mean_default_cost']:.2f}", f"{result['mean_selected_cost']:.2f}",
                  change_cell(result['mean_paired_change_pct']), f"{result['default_sd']:.2f}",
                  f"{result['selected_sd']:.2f}"]
        validation_table.append(" & ".join(values) + r" \\")
    validation_table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (args.validation / "summary_table.tex").write_text("\n".join(validation_table) + "\n")
    analysis = {"selected": selected, "tuning": selection, "validation_cases": case_results,
                "pooled_validation_change_pct": float(np.mean([r["mean_paired_change_pct"] for r in case_results])),
                "validation_instance_mean_wins": sum(r["mean_selected_cost"] < r["mean_default_cost"] for r in case_results),
                "validation_paired_wins": sum(r["paired_wins"] for r in case_results)}
    for tasks in (100, 200):
        cases = [r for r in case_results if r["tasks"] == tasks]
        analysis[f"validation_{tasks}_change_pct"] = float(np.mean([r["mean_paired_change_pct"] for r in cases]))
    (args.validation / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
