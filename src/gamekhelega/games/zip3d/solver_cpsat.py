"""CP-SAT model for 3D Zip (OR-Tools, optional extra ``gamekhelega[zip3d]``).

Arc literal per open directed edge u->v, plus a dummy node D with arcs D->start and
end->D fixed true; AddCircuit makes the arcs one Hamiltonian cycle through D, i.e. a
Hamiltonian path start..end. rank[c] in [0, N-1] (rank[v] = rank[u] + 1 on used arcs)
orders the waypoints: rank[w_k] < rank[w_{k+1}]. Solutions are counted by re-solving
with a blocking clause on the previous solution's arcs (up to `limit`).
"""
from __future__ import annotations

import time

from .model import ZipPuzzle, neighbours, edge


def available() -> bool:
    try:
        import ortools  # noqa: F401
        return True
    except ImportError:
        return False


def cpsat_count(p: ZipPuzzle, limit: int = 2, time_limit: float | None = None,
                workers: int = 8) -> tuple[int | None, list[list[int]]]:
    """(count capped at limit, solutions as cell lists). count None = undecided (timeout)."""
    from ortools.sat.python import cp_model

    n, N = p.n, p.n ** 3
    t0 = time.monotonic()
    s, e = p.cell_of(1), p.cell_of(p.K)
    m = cp_model.CpModel()
    D = N
    arcs, lit = [], {}
    for u in range(N):
        for _, v in neighbours(n, u):
            if edge(u, v) in p.walls or v == s or u == e:
                continue
            x = m.NewBoolVar(f"a{u}_{v}")
            lit[(u, v)] = x
            arcs.append((u, v, x))
    arcs.append((D, s, m.NewConstant(1)))
    arcs.append((e, D, m.NewConstant(1)))
    m.AddCircuit(arcs)
    rank = [m.NewIntVar(0, N - 1, f"r{c}") for c in range(N)]
    m.Add(rank[s] == 0)
    m.Add(rank[e] == N - 1)
    for (u, v), x in lit.items():
        m.Add(rank[v] == rank[u] + 1).OnlyEnforceIf(x)
    order = sorted(p.waypoints.items(), key=lambda t: t[1])
    for (a, _), (b, _) in zip(order, order[1:]):
        m.Add(rank[a] < rank[b])
    sols: list[list[int]] = []
    while len(sols) < limit:
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = workers
        if time_limit is not None:
            left = time_limit - (time.monotonic() - t0)
            if left <= 0:
                return None, sols
            solver.parameters.max_time_in_seconds = left
        st = solver.Solve(m)
        if st == cp_model.INFEASIBLE:
            break
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, sols
        r = [solver.Value(x) for x in rank]
        path = [0] * N
        for c, k in enumerate(r):
            path[k] = c
        sols.append(path)
        used = [x for x in lit.values() if solver.Value(x)]
        m.AddBoolOr([x.Not() for x in used])
    return len(sols), sols
