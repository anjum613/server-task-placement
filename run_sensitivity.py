"""One-factor-at-a-time sensitivity analysis on the 20-task seed-11 case."""

from __future__ import annotations

import csv
from pathlib import Path

from scheduler import Instance, generate_instance, solve_linear


def main() -> None:
    original = generate_instance(20, 4, 11)
    base = original.to_dict()
    scenarios = [("baseline", 1.0, 1.0, 1.0, 25.0),
                 ("fixed cost x 0.5", 0.5, 1.0, 1.0, 25.0),
                 ("fixed cost x 2", 2.0, 1.0, 1.0, 25.0),
                 ("capacity x 0.9", 1.0, 0.9, 1.0, 25.0),
                 ("capacity x 1.1", 1.0, 1.1, 1.0, 25.0),
                 ("task demand x 0.9", 1.0, 1.0, 0.9, 25.0),
                 ("task demand x 1.1", 1.0, 1.0, 1.1, 25.0),
                 ("balance weight 0", 1.0, 1.0, 1.0, 0.0),
                 ("balance weight 50", 1.0, 1.0, 1.0, 50.0)]
    rows = []
    for label, fixed_factor, capacity_factor, demand_factor, weight in scenarios:
        data = dict(base)
        data["task_cpu"] = [v * demand_factor for v in base["task_cpu"]]
        data["task_mem"] = [v * demand_factor for v in base["task_mem"]]
        data["fixed_cost"] = [v * fixed_factor for v in base["fixed_cost"]]
        data["server_cpu"] = [v * capacity_factor for v in base["server_cpu"]]
        data["server_mem"] = [v * capacity_factor for v in base["server_mem"]]
        data["balance_weight"] = weight
        inst = Instance.from_dict(data)
        result = solve_linear(inst, 10.0)
        row = {"scenario": label, "status": result["solver_status"],
               "runtime_s": result["runtime"], "objective": result["objective"],
               "bound": result["bound"], "gap": result["gap"]}
        if result["assignment"] is not None:
            row.update(result["metrics"])
        rows.append(row)
    out = Path(__file__).parent / "results" / "sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
