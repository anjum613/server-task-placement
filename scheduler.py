"""Static task-to-server allocation: MILP, LP bound, greedy, and genetic search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Instance:
    task_cpu: np.ndarray
    task_mem: np.ndarray
    server_cpu: np.ndarray
    server_mem: np.ndarray
    fixed_cost: np.ndarray
    assignment_cost: np.ndarray
    balance_weight: float = 25.0

    @property
    def n(self) -> int:
        return len(self.task_cpu)

    @property
    def m(self) -> int:
        return len(self.server_cpu)

    def validate(self) -> None:
        if self.n == 0 or self.m == 0:
            raise ValueError("At least one task and server are required")
        if len(self.task_mem) != self.n or any(len(v) != self.m for v in
                                               (self.server_mem, self.fixed_cost)):
            raise ValueError("Resource and cost vector lengths do not match")
        if self.assignment_cost.shape != (self.n, self.m):
            raise ValueError("assignment_cost must have shape (tasks, servers)")
        for v in (self.task_cpu, self.task_mem, self.server_cpu,
                  self.server_mem, self.fixed_cost, self.assignment_cost):
            if not np.all(np.isfinite(v)) or np.any(v <= 0):
                raise ValueError("All demands, capacities and costs must be positive and finite")
        if not np.isfinite(self.balance_weight) or self.balance_weight < 0:
            raise ValueError("balance_weight must be finite and nonnegative")

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in
                ("task_cpu", "task_mem", "server_cpu", "server_mem",
                 "fixed_cost", "assignment_cost")} | {"balance_weight": self.balance_weight}

    @classmethod
    def from_dict(cls, data: dict) -> "Instance":
        keys = ("task_cpu", "task_mem", "server_cpu", "server_mem",
                "fixed_cost", "assignment_cost")
        inst = cls(*(np.asarray(data[k], dtype=float) for k in keys),
                   float(data.get("balance_weight", 25.0)))
        inst.validate()
        return inst


def generate_instance(n: int, m: int, seed: int, balance_weight: float = 25.0) -> Instance:
    """Generate synthetic, heterogeneous data; no real cloud measurements are implied."""
    rng = np.random.default_rng(seed)
    cpu = rng.integers(1, 6, n).astype(float)
    mem = rng.integers(2, 9, n).astype(float)
    server_cpu = rng.integers(30, 41, m).astype(float)
    server_mem = rng.integers(57, 76, m).astype(float)
    fixed = rng.integers(65, 116, m).astype(float)
    cpu_rate = rng.uniform(0.55, 1.55, m)
    mem_rate = rng.uniform(0.10, 0.40, m)
    costs = 1.5 + cpu[:, None] * cpu_rate + mem[:, None] * mem_rate
    costs += rng.uniform(0, 1, (n, m))
    inst = Instance(cpu, mem, server_cpu, server_mem, fixed, costs, balance_weight)
    inst.validate()
    return inst


def evaluate(inst: Instance, assignment: np.ndarray) -> dict:
    """Independent objective and hard-constraint check used for every method."""
    raw = np.asarray(assignment, dtype=float)
    if not np.all(np.isfinite(raw)) or np.any(np.abs(raw - np.rint(raw)) > 1e-7):
        raise ValueError("Server indices must be finite integers")
    a = raw.astype(int)
    if a.shape != (inst.n,) or np.any(a < 0) or np.any(a >= inst.m):
        raise ValueError("Each task must name exactly one valid server")
    cpu = np.bincount(a, weights=inst.task_cpu, minlength=inst.m)
    mem = np.bincount(a, weights=inst.task_mem, minlength=inst.m)
    if np.any(cpu > inst.server_cpu + 1e-7) or np.any(mem > inst.server_mem + 1e-7):
        raise ValueError("CPU or memory capacity exceeded")
    active = np.bincount(a, minlength=inst.m) > 0
    fixed = float(inst.fixed_cost[active].sum())
    running = float(inst.assignment_cost[np.arange(inst.n), a].sum())
    max_cpu = float(np.max(cpu / inst.server_cpu))
    balance = inst.balance_weight * max_cpu
    return dict(objective=fixed + running + balance, fixed=fixed, running=running,
                balance=balance, active=int(active.sum()), max_cpu_util=max_cpu,
                max_mem_util=float(np.max(mem / inst.server_mem)),
                mean_active_cpu_util=float(np.mean((cpu / inst.server_cpu)[active])),
                mean_active_mem_util=float(np.mean((mem / inst.server_mem)[active])))


def greedy(inst: Instance) -> tuple[np.ndarray | None, float]:
    """First fit decreasing by the larger normalized resource demand."""
    start = perf_counter()
    order = np.argsort(-np.maximum(inst.task_cpu / np.mean(inst.server_cpu),
                                   inst.task_mem / np.mean(inst.server_mem)), kind="stable")
    cpu = np.zeros(inst.m)
    mem = np.zeros(inst.m)
    a = np.full(inst.n, -1, dtype=int)
    for i in order:
        feasible = np.flatnonzero((cpu + inst.task_cpu[i] <= inst.server_cpu) &
                                  (mem + inst.task_mem[i] <= inst.server_mem))
        if len(feasible) == 0:
            return None, perf_counter() - start
        j = int(feasible[0])
        a[i] = j
        cpu[j] += inst.task_cpu[i]
        mem[j] += inst.task_mem[i]
    return a, perf_counter() - start


def solve_linear(inst: Instance, time_limit: float, relax: bool = False) -> dict:
    """SciPy/HiGHS MILP; relax=True gives an LP lower bound."""
    start = perf_counter()
    n, m = inst.n, inst.m
    nx = n * m
    ny = m
    nz = nx + ny
    nv = nz + 1
    c = np.concatenate([inst.assignment_cost.ravel(), inst.fixed_cost,
                        [inst.balance_weight]])
    # Rows: task assignment, CPU, memory, normalized CPU peak, activation.
    matrix = lil_matrix((n + 4 * m, nv), dtype=float)
    lower = np.full(n + 4 * m, -np.inf)
    upper = np.zeros(n + 4 * m)
    for i in range(n):
        matrix[i, i * m:(i + 1) * m] = 1
        lower[i] = upper[i] = 1
        for j in range(m):
            col = i * m + j
            matrix[n + j, col] = inst.task_cpu[i]
            matrix[n + m + j, col] = inst.task_mem[i]
            matrix[n + 2 * m + j, col] = inst.task_cpu[i] / inst.server_cpu[j]
            matrix[n + 3 * m + j, col] = 1
    for j in range(m):
        matrix[n + j, nx + j] = -inst.server_cpu[j]
        matrix[n + m + j, nx + j] = -inst.server_mem[j]
        matrix[n + 2 * m + j, nz] = -1
        matrix[n + 3 * m + j, nx + j] = -1
        lower[n + 3 * m + j] = 0
        upper[n + 3 * m + j] = np.inf
    bounds = Bounds(np.zeros(nv), np.r_[np.ones(nx + ny), 1.0])
    integrality = np.r_[np.ones(nx + ny), 0].astype(int)
    if relax:
        integrality[:] = 0
    res = milp(c, integrality=integrality, bounds=bounds,
               constraints=LinearConstraint(matrix.tocsr(), lower, upper),
               options={"time_limit": float(time_limit), "mip_rel_gap": 1e-6})
    elapsed = perf_counter() - start
    out = {"runtime": elapsed, "solver_status": int(res.status),
           "solver_message": str(res.message), "bound": None, "gap": None,
           "assignment": None, "objective": None, "solver_objective": None}
    if res.x is not None and res.fun is not None and np.isfinite(res.fun):
        out["solver_objective"] = float(res.fun)
        if not relax:
            x = res.x[:nx].reshape(n, m)
            a = np.argmax(x, axis=1)
            if np.all(np.max(x, axis=1) >= 1 - 1e-5):
                try:
                    out["metrics"] = evaluate(inst, a)
                    out["assignment"] = a
                    out["objective"] = out["metrics"]["objective"]
                except ValueError:
                    pass
        else:
            out["objective"] = float(res.fun)
    for src, dst in (("mip_dual_bound", "bound"), ("mip_gap", "gap")):
        val = getattr(res, src, None)
        if val is not None and np.isfinite(val):
            out[dst] = float(val)
    if not relax and out["objective"] is not None and out["bound"] is not None:
        out["gap"] = max(0.0, (out["objective"] - out["bound"]) / out["objective"])
    return out


def _decode(inst: Instance, rng: np.random.Generator,
            preferred: np.ndarray | None = None, *, retention: float = 0.83,
            noise: float = 12.0, order_jitter: float = 0.025) -> np.ndarray | None:
    """Feasible assignment repair with a randomized incremental-cost choice."""
    n, m = inst.n, inst.m
    cpu = np.zeros(m)
    mem = np.zeros(m)
    a = np.full(n, -1, dtype=int)
    sizes = np.maximum(inst.task_cpu / np.mean(inst.server_cpu),
                       inst.task_mem / np.mean(inst.server_mem))
    order = np.argsort(-(sizes + rng.uniform(0, order_jitter, n)))
    for i in order:
        feasible = (cpu + inst.task_cpu[i] <= inst.server_cpu) & (
            mem + inst.task_mem[i] <= inst.server_mem)
        if not np.any(feasible):
            return None
        p = -1 if preferred is None else int(preferred[i])
        if 0 <= p < m and feasible[p] and rng.random() < retention:
            j = p
        else:
            next_peak = np.maximum(np.max(cpu / inst.server_cpu),
                                   (cpu + inst.task_cpu[i]) / inst.server_cpu)
            score = inst.assignment_cost[i] + inst.fixed_cost * (cpu == 0)
            score += inst.balance_weight * (
                next_peak - np.max(cpu / inst.server_cpu))
            score += rng.uniform(0, noise, m)
            score[~feasible] = np.inf
            j = int(np.argmin(score))
        a[i] = j
        cpu[j] += inst.task_cpu[i]
        mem[j] += inst.task_mem[i]
    return a


def _improve(inst: Instance, a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One pass of feasible single-task moves, in random task order."""
    best = a.copy()
    cpu = np.bincount(best, weights=inst.task_cpu, minlength=inst.m)
    mem = np.bincount(best, weights=inst.task_mem, minlength=inst.m)
    counts = np.bincount(best, minlength=inst.m)
    for i in rng.permutation(inst.n):
        old = best[i]
        candidates = np.flatnonzero((cpu + inst.task_cpu[i] <= inst.server_cpu) &
                                    (mem + inst.task_mem[i] <= inst.server_mem))
        for j in candidates:
            if j == old:
                continue
            fixed_delta = (inst.fixed_cost[j] if counts[j] == 0 else 0) - (
                inst.fixed_cost[old] if counts[old] == 1 else 0)
            running_delta = inst.assignment_cost[i, j] - inst.assignment_cost[i, old]
            peak_cpu = cpu.copy()
            peak_cpu[old] -= inst.task_cpu[i]
            peak_cpu[j] += inst.task_cpu[i]
            balance_delta = inst.balance_weight * (
                np.max(peak_cpu / inst.server_cpu) - np.max(cpu / inst.server_cpu))
            delta = fixed_delta + running_delta + balance_delta
            if delta < -1e-8:
                best[i] = j
                cpu = peak_cpu
                mem[old] -= inst.task_mem[i]
                mem[j] += inst.task_mem[i]
                counts[old] -= 1
                counts[j] += 1
                break
    return best


@dataclass(frozen=True)
class GAConfig:
    """Fixed settings for one run. Probabilities are fractions, noise is in cost units."""
    population_size: int = 24
    selection: str = "tournament"
    tournament_size: int = 3
    crossover_rate: float = 1.0
    mutation_rate: float | None = None
    retention: float = 0.83
    decoder_noise: float = 12.0
    order_jitter: float = 0.025

    def validate(self) -> None:
        for name, minimum in (("population_size", 3), ("tournament_size", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if self.selection not in ("tournament", "rank"):
            raise ValueError("selection must be tournament or rank")
        probabilities = (self.crossover_rate, self.retention)
        if self.mutation_rate is not None:
            probabilities += (self.mutation_rate,)
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
            raise ValueError("Crossover, mutation and retention must be between zero and one")
        if any(not np.isfinite(v) or v < 0 for v in
               (self.decoder_noise, self.order_jitter)):
            raise ValueError("Decoder noise and order jitter must be finite and nonnegative")

    def resolved_mutation_rate(self, n: int) -> float:
        return min(1.0, max(0.025, 2 / n)) if self.mutation_rate is None else self.mutation_rate


def _select_parent(population: list[tuple[float, np.ndarray]],
                   rng: np.random.Generator, config: GAConfig) -> np.ndarray:
    """The population is sorted by increasing cost; tournament draws replace samples."""
    count = len(population)
    if config.selection == "rank":
        weights = np.arange(count, 0, -1, dtype=float)
        index = int(rng.choice(count, p=weights / weights.sum()))
        return population[index][1]
    ids = rng.integers(0, count, min(config.tournament_size, count))
    return min((population[k] for k in ids), key=lambda item: item[0])[1]


def _recombine(p1: np.ndarray, p2: np.ndarray, m: int,
               rng: np.random.Generator, config: GAConfig) -> np.ndarray:
    """Uniform crossover, then independent random-reset mutation of preferred indices."""
    if config.crossover_rate == 1 or (
            config.crossover_rate > 0 and rng.random() < config.crossover_rate):
        preferred = np.where(rng.random(len(p1)) < 0.5, p1, p2).copy()
    else:
        preferred = p1.copy()
    mut = rng.random(len(p1)) < config.resolved_mutation_rate(len(p1))
    # A reset can choose the old value again; this is an attempted mutation rate.
    preferred[mut] = rng.integers(0, m, int(mut.sum()))
    return preferred


def genetic(inst: Instance, time_limit: float, seed: int,
            population_size: int = 24, *, config: GAConfig | None = None) -> dict:
    """Feasible GA with explicit settings, two elites and single-task local improvement.

    Initialization and search share the same wall-time allowance. A repair or local
    pass already in progress is completed, so measured time can exceed the limit.
    """
    settings = config or GAConfig(population_size=population_size)
    settings.validate()
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    population_size = settings.population_size
    rng = np.random.default_rng(seed)
    start = perf_counter()
    deadline = start + time_limit
    population: list[tuple[float, np.ndarray]] = []
    greedy_a, _ = greedy(inst)
    if greedy_a is not None:
        population.append((evaluate(inst, greedy_a)["objective"], greedy_a))
    attempts = 0
    while len(population) < population_size and perf_counter() < deadline and attempts < 5 * population_size:
        attempts += 1
        a = _decode(inst, rng, rng.integers(0, inst.m, inst.n),
                    retention=settings.retention, noise=settings.decoder_noise,
                    order_jitter=settings.order_jitter)
        if a is not None:
            population.append((evaluate(inst, a)["objective"], a))
    initialization_s = perf_counter() - start
    initial_population_size = len(population)
    diagnostics = {"initialization_s": initialization_s,
                   "initial_population_size": initial_population_size,
                   "initialization_attempts": attempts,
                   "initialization_failures": attempts - initial_population_size + int(greedy_a is not None),
                   "settings": asdict(settings) | {
                       "effective_mutation_rate": settings.resolved_mutation_rate(inst.n)}}
    if not population:
        return {"assignment": None, "runtime": perf_counter() - start,
                "generations": 0, "status": "no feasible solution found", **diagnostics}
    population.sort(key=lambda item: item[0])
    best_cost, best_a = population[0][0], population[0][1].copy()
    initial_best_objective = best_cost
    trace = [{"time_s": perf_counter() - start, "objective": best_cost}]
    generations = complete_generations = offspring_attempts = offspring_accepted = 0
    while perf_counter() < deadline and len(population) >= 2:
        next_population = population[:2]
        while len(next_population) < population_size and perf_counter() < deadline:
            offspring_attempts += 1
            p1 = _select_parent(population, rng, settings)
            p2 = _select_parent(population, rng, settings)
            preferred = _recombine(p1, p2, inst.m, rng, settings)
            child = _decode(inst, rng, preferred, retention=settings.retention,
                            noise=settings.decoder_noise, order_jitter=settings.order_jitter)
            if child is None:
                continue
            child = _improve(inst, child, rng)
            cost = evaluate(inst, child)["objective"]
            next_population.append((cost, child))
            offspring_accepted += 1
            if cost < best_cost:
                best_cost, best_a = cost, child.copy()
                trace.append({"time_s": perf_counter() - start, "objective": best_cost})
        if len(next_population) <= 2:
            break
        population = sorted(next_population, key=lambda item: item[0])
        generations += 1
        complete_generations += int(len(next_population) == population_size)
    return {"assignment": best_a, "metrics": evaluate(inst, best_a),
            "runtime": perf_counter() - start, "generations": generations,
            "complete_generations": complete_generations,
            "offspring_attempts": offspring_attempts,
            "offspring_accepted": offspring_accepted,
            "offspring_failures": offspring_attempts - offspring_accepted,
            "initial_best_objective": initial_best_objective,
            "time_to_best_s": trace[-1]["time_s"], "trace": trace,
            "status": "feasible", **diagnostics}


def exhaustive_check() -> dict:
    """Independent enumeration verifies the MILP and objective on a small case."""
    inst = Instance.from_dict({
        "task_cpu": [4, 5, 6, 3, 2], "task_mem": [6, 5, 7, 4, 3],
        "server_cpu": [9, 10, 12], "server_mem": [12, 13, 14],
        "fixed_cost": [22, 25, 30],
        "assignment_cost": [[5, 7, 6], [8, 4, 6], [7, 6, 5],
                            [4, 6, 5], [6, 5, 4]],
        "balance_weight": 12.0})
    candidates = []
    for tup in product(range(inst.m), repeat=inst.n):
        try:
            candidates.append((evaluate(inst, np.asarray(tup))["objective"], tup))
        except ValueError:
            continue
    exact = min(candidates)
    milp_result = solve_linear(inst, 20)
    if milp_result["assignment"] is None:
        raise AssertionError("MILP did not return a small-case solution")
    measured = evaluate(inst, milp_result["assignment"])["objective"]
    if abs(measured - exact[0]) > 1e-5 or abs(measured - milp_result["objective"]) > 1e-5:
        raise AssertionError("Small-case enumeration, MILP and independent check disagree")
    return {"feasible_assignments": len(candidates), "optimum": exact[0],
            "assignment": list(exact[1])}
