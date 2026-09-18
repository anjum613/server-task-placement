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

