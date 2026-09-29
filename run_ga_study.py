"""Controlled GA setting study on larger instances, with a separate validation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy

from scheduler import GAConfig, Instance, evaluate, generate_instance, genetic


def study_configurations() -> dict[str, GAConfig]:
    base = GAConfig()
    return {
        "baseline": base,
        "population_12": replace(base, population_size=12),
        "population_48": replace(base, population_size=48),
        "tournament_2": replace(base, tournament_size=2),
        "tournament_5": replace(base, tournament_size=5),
        "rank_selection": replace(base, selection="rank"),
        "crossover_0": replace(base, crossover_rate=0.0),
        "crossover_06": replace(base, crossover_rate=0.6),
        "mutation_0": replace(base, mutation_rate=0.0),
        "mutation_001": replace(base, mutation_rate=0.01),
        "mutation_008": replace(base, mutation_rate=0.08),
    }


def summarise_study(rows: list[dict]) -> list[dict]:
    """Equal weight to each case; paired change uses the same case and random seed."""
    baseline = {(r["case"], r["repeat"]): r["objective"] for r in rows
                if r["configuration"] == "baseline" and r["status"] == "feasible"}
    summary = []
    for name in dict.fromkeys(r["configuration"] for r in rows):
        for tasks in sorted({r["tasks"] for r in rows}):
            group = [r for r in rows if r["configuration"] == name and r["tasks"] == tasks]
            feasible = [r for r in group if r["status"] == "feasible"]
            if not feasible:
                continue
            deviations = []
            case_means = []
            for case in dict.fromkeys(r["case"] for r in feasible):
                costs = [r["objective"] for r in feasible if r["case"] == case]
                case_means.append(float(np.mean(costs)))
                deviations.append(float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0)
            changes = [100 * (r["objective"] / baseline[(r["case"], r["repeat"])] - 1)
                       for r in feasible if (r["case"], r["repeat"]) in baseline]
            summary.append({
                "configuration": name, "tasks": tasks, "runs": len(group),
                "feasible": len(feasible), "mean_objective": float(np.mean(case_means)),
                "mean_within_instance_sd": float(np.mean(deviations)),
                "mean_paired_change_pct": float(np.mean(changes)) if changes else None,
                "paired_change_sd_pct": float(np.std(changes, ddof=1)) if len(changes) > 1 else 0.0,
                "mean_complete_generations": float(np.mean([r["complete_generations"] for r in feasible])),
                "mean_offspring": float(np.mean([r["offspring_accepted"] for r in feasible])),
                "mean_initialization_s": float(np.mean([r["initialization_s"] for r in feasible])),
                "mean_runtime_s": float(np.mean([r["runtime_s"] for r in feasible])),
                "mean_time_to_best_s": float(np.mean([r["time_to_best_s"] for r in feasible])),
            })
    return summary


def verify_study(root: Path) -> int:
    """Recompute all costs and both resource constraints from saved allocations."""
    with (root / "raw_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    instances: dict[str, Instance] = {}
    count = 0
    for row in rows:
        case = row["case"]
        if case not in instances:
            instances[case] = Instance.from_dict(json.loads(
                (root / "instances" / f"{case}.json").read_text()))
        solution = root / "solutions" / row["solution_file"]
        if row["status"] != "feasible":
            assert not solution.exists(), solution
            continue
        saved = json.loads(solution.read_text())
        actual = evaluate(instances[case], np.asarray(saved["assignment"]))
        for key in actual:
            assert abs(actual[key] - float(row[key])) < 1e-6, (solution, key)
        assert saved["settings"]["effective_mutation_rate"] == float(row["effective_mutation_rate"])
        trace = saved["trace"]
        assert abs(trace[-1]["objective"] - actual["objective"]) < 1e-6, solution
        assert all(b["objective"] < a["objective"] for a, b in zip(trace, trace[1:])), solution
        assert all(b["time_s"] >= a["time_s"] for a, b in zip(trace, trace[1:])), solution
        assert trace[-1]["time_s"] <= float(row["runtime_s"]), solution
        count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tuning", "validation"), default="tuning")
    parser.add_argument("--configs", nargs="+", choices=tuple(study_configurations()))
    parser.add_argument("--sizes", nargs="+", default=["100x12", "200x24"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="only verify a previously saved GA study")
    args = parser.parse_args(argv)
    if args.verify:
        print(f"Verified {verify_study(args.verify)} GA study allocations")
        return
    if not np.isfinite(args.budget) or args.budget <= 0 or args.repetitions <= 0:
        parser.error("budget and repetitions must be positive")
    if args.phase == "validation" and not args.configs:
        parser.error("validation requires --configs baseline and the setting selected on tuning data")
    settings = study_configurations()
    names = args.configs or list(settings)
    if "baseline" not in names:
        parser.error("include baseline to obtain paired comparisons")
    seeds = args.seeds or ([11, 22, 33] if args.phase == "tuning" else [44, 55, 66])
    root = args.output or Path(__file__).parent / f"ga_{args.phase}"
    (root / "instances").mkdir(parents=True, exist_ok=True)
    (root / "solutions").mkdir(parents=True, exist_ok=True)
    instances = {}
    blocks = []
    for size in args.sizes:
        n, m = map(int, size.lower().split("x"))
        if n <= 0 or m <= 0:
            parser.error("task and server counts must be positive")
        for seed in seeds:
            inst = generate_instance(n, m, seed)
            case = f"n{n}_m{m}_s{seed}"
            instances[case] = inst
            (root / "instances" / f"{case}.json").write_text(json.dumps(inst.to_dict(), indent=2) + "\n")
            blocks.extend((case, seed, rep) for rep in range(1, args.repetitions + 1))
    order_rng = np.random.default_rng(args.order_seed)
    order_rng.shuffle(blocks)
    jobs = [(case, seed, rep, str(name)) for case, seed, rep in blocks
            for name in order_rng.permutation(names)]
    source = Path(__file__).read_bytes()
    core_path = Path(__file__).with_name("scheduler.py")
    metadata = {
        "phase": args.phase, "budget_s": args.budget, "repetitions": args.repetitions,
        "instance_seeds": seeds, "sizes": args.sizes, "order_seed": args.order_seed,
        "run_order": jobs, "configuration_settings": {k: asdict(settings[k]) for k in names},
        "ga_seed_rule": "instance_seed * 1000 + repeat + 16",
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest() if core_path.exists() else None,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with (root / "raw_results.csv").open("w", newline="") as stream:
        writer = None
        for index, (case, seed, rep, name) in enumerate(jobs, start=1):
            inst = instances[case]
            ga_seed = seed * 1000 + rep + 16
            result = genetic(inst, args.budget, ga_seed, config=settings[name])
            solution_file = f"{case}_{name}_r{rep}.json"
            row = {
                "phase": args.phase, "run_order": index, "case": case, "tasks": inst.n,
                "servers": inst.m, "instance_seed": seed, "repeat": rep, "ga_seed": ga_seed,
                "configuration": name, "status": result["status"], "budget_s": args.budget,
                "runtime_s": result["runtime"], "solution_file": solution_file,
                **result["settings"],
                **{k: result.get(k, 0) for k in ("generations", "complete_generations",
                    "offspring_attempts", "offspring_accepted", "offspring_failures",
                    "initialization_s", "initial_population_size", "initialization_attempts",
                    "initialization_failures", "initial_best_objective", "time_to_best_s")},
                **{k: None for k in ("objective", "fixed", "running", "balance", "active",
                    "max_cpu_util", "max_mem_util", "mean_active_cpu_util", "mean_active_mem_util")},
            }
            if result["assignment"] is not None:
                row.update(evaluate(inst, result["assignment"]))
                (root / "solutions" / solution_file).write_text(json.dumps({
                    "assignment": result["assignment"].tolist(), "trace": result["trace"],
                    "settings": result["settings"], "ga_seed": ga_seed}, indent=2) + "\n")
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            cost_text = f"{row['objective']:.2f}" if row['objective'] is not None else result["status"]
            print(f"{index}/{len(jobs)} {case} r{rep} {name}: {cost_text} "
                  f"({row['complete_generations']} full generations)", flush=True)
    summary = summarise_study(rows)
    if summary:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(f"Verified {verify_study(root)} saved allocations; study saved in {root}")


if __name__ == "__main__":
    main()
