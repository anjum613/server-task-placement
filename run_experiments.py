"""Run reproducible benchmarks and save raw results, instances, and solutions."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from scheduler import (GAConfig, Instance, evaluate, exhaustive_check, generate_instance,
                       genetic, greedy, solve_linear)


def record(case: str, instance_seed: int | None, method: str, repeat: int | None,
           result: dict, inst: Instance) -> dict:
    a = result.get("assignment")
    row = {"case": case, "tasks": inst.n, "servers": inst.m,
           "instance_seed": instance_seed, "method": method, "repeat": repeat,
           "status": result.get("status", result.get("solver_status", "")),
           "runtime_s": result["runtime"], "objective": "",
           "solver_objective": result.get("solver_objective"), "fixed": "",
           "running": "", "balance": "", "active": "", "max_cpu_util": "",
           "max_mem_util": "", "mean_active_cpu_util": "",
           "mean_active_mem_util": "", "bound": result.get("bound"),
           "gap": result.get("gap"), "generations": result.get("generations")}
    if a is not None:
        row.update(evaluate(inst, a))
    elif method == "LP relaxation":
        row["objective"] = result.get("objective")
    return row


def run_one(inst: Instance, case: str, seed: int | None, budget: float,
            repetitions: int, outdir: Path, config: GAConfig | None = None) -> list[dict]:
    (outdir / "instances").mkdir(parents=True, exist_ok=True)
    (outdir / "solutions").mkdir(parents=True, exist_ok=True)
    (outdir / "instances" / f"{case}.json").write_text(
        json.dumps(inst.to_dict(), indent=2) + "\n")
    rows = []
    g, elapsed = greedy(inst)
    gr = {"assignment": g, "runtime": elapsed,
          "status": "feasible" if g is not None else "infeasible construction"}
    rows.append(record(case, seed, "Greedy", None, gr, inst))
    if g is not None:
        (outdir / "solutions" / f"{case}_greedy.json").write_text(json.dumps(g.tolist()) + "\n")
    lp = solve_linear(inst, budget, relax=True)
    rows.append(record(case, seed, "LP relaxation", None, lp, inst))
    exact = solve_linear(inst, budget)
    rows.append(record(case, seed, "MILP", None, exact, inst))
    if exact["assignment"] is not None:
        (outdir / "solutions" / f"{case}_milp.json").write_text(
            json.dumps(exact["assignment"].tolist()) + "\n")
    for rep in range(repetitions):
        ga = genetic(inst, budget, seed=(seed or 0) * 1000 + rep + 17, config=config)
        rows.append(record(case, seed, "GA", rep + 1, ga, inst))
        if ga["assignment"] is not None:
            (outdir / "solutions" / f"{case}_ga_{rep+1}.json").write_text(
                json.dumps(ga["assignment"].tolist()) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--budget", type=float, default=1.0,
                        help="seconds per MILP, LP, or GA run")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--sizes", type=str, nargs="+",
                        default=["20x4", "50x8", "100x12", "200x24"])
    parser.add_argument("--instance", type=Path,
                        help="run a custom JSON instance instead of generated cases")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--selection", choices=("tournament", "rank"), default="tournament")
    parser.add_argument("--tournament", type=int, default=3)
    parser.add_argument("--crossover", type=float, default=1.0)
    parser.add_argument("--mutation", type=float, default=None,
                        help="per-task reset probability; default min(1, max(0.025, 2/n))")
    parser.add_argument("--retention", type=float, default=0.83)
    parser.add_argument("--decoder-noise", type=float, default=12.0)
    args = parser.parse_args()
    if not np.isfinite(args.budget) or args.budget <= 0 or args.repetitions <= 0:
        parser.error("budget and repetitions must be positive")
    config = GAConfig(population_size=args.population, selection=args.selection,
                      tournament_size=args.tournament, crossover_rate=args.crossover,
                      mutation_rate=args.mutation, retention=args.retention,
                      decoder_noise=args.decoder_noise)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    check = exhaustive_check()
    rows: list[dict] = []
    if args.instance:
        inst = Instance.from_dict(json.loads(args.instance.read_text()))
        rows += run_one(inst, args.instance.stem, None, args.budget,
                        args.repetitions, args.output, config)
    else:
        for size in args.sizes:
            n, m = map(int, size.lower().split("x"))
            for seed in args.seeds:
                inst = generate_instance(n, m, seed)
                case = f"n{n}_m{m}_s{seed}"
                print(f"Running {case}", flush=True)
                rows += run_one(inst, case, seed, args.budget,
                                args.repetitions, args.output, config)
    with (args.output / "raw_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"budget_s": args.budget, "ga_repetitions": args.repetitions,
                "seeds": args.seeds if not args.instance else None,
                "sizes": args.sizes if not args.instance else None,
                "small_case_check": check,
                "python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "scipy": scipy.__version__,
                "ga_settings": asdict(config)}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(rows)} rows to {args.output / 'raw_results.csv'}")


if __name__ == "__main__":
    main()
