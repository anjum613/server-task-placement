"""Recheck every saved solution and every reported bound against saved data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scheduler import Instance, evaluate, exhaustive_check


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
root = parser.parse_args().results
rows = pd.read_csv(root / "raw_results.csv")
checked = 0
for case, group in rows.groupby("case"):
    inst = Instance.from_dict(json.loads((root / "instances" / f"{case}.json").read_text()))
    feasible = []
    for row in group.itertuples():
        if row.method == "LP relaxation":
            continue
        suffix = ("greedy" if row.method == "Greedy" else
                  "milp" if row.method == "MILP" else f"ga_{int(row.repeat)}")
        file = root / "solutions" / f"{case}_{suffix}.json"
        if pd.isna(row.objective):
            assert not file.exists(), file
            continue
        assignment = np.asarray(json.loads(file.read_text()))
        actual = evaluate(inst, assignment)
        for key in ("objective", "fixed", "running", "balance", "active",
                    "max_cpu_util", "max_mem_util"):
            assert abs(actual[key] - getattr(row, key)) < 1e-6, (file, key)
        feasible.append(actual["objective"])
        checked += 1
    lp = group[group.method == "LP relaxation"].iloc[0]
    mip = group[group.method == "MILP"].iloc[0]
    assert lp.objective <= min(feasible) + 1e-5, case
    assert mip.bound <= min(feasible) + 1e-5, case
    if str(mip.status) == "0":
        assert abs(mip.objective - mip.bound) < 1e-4, case

small = exhaustive_check()
print(f"Verified {checked} saved feasible allocations across {rows.case.nunique()} instances")
print(f"Small case: {small['feasible_assignments']} feasible assignments, optimum {small['optimum']}")
