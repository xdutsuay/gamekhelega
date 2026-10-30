"""CP-SAT reference solver (benchmark only): one-hot bools + AddExactlyOne, stop at `limit`."""
import numpy as np
from ortools.sat.python import cp_model


def cpsat_count(puzzle, rules, limit=2, workers=1, time_limit=None):
    """Returns solutions found (capped at limit); -1 if time ran out before deciding."""
    n, N = rules.n, rules.ncells
    m = cp_model.CpModel()
    x = [[m.NewBoolVar("") for _ in range(n)] for _ in range(N)]
    for c in range(N):
        m.AddExactlyOne(x[c])
    for row in rules.units:
        for v in range(n):
            m.AddExactlyOne([x[int(c)][v] for c in row])
    g = np.asarray(puzzle).ravel()
    for c in np.flatnonzero(g):
        m.Add(x[c][int(g[c]) - 1] == 1)

    class Cb(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__(); self.k = 0
        def on_solution_callback(self):
            self.k += 1
            if self.k >= limit:
                self.StopSearch()

    s = cp_model.CpSolver()
    s.parameters.enumerate_all_solutions = True
    s.parameters.num_workers = workers
    if time_limit:
        s.parameters.max_time_in_seconds = float(time_limit)
    cb = Cb()
    st = s.Solve(m, cb)
    if cb.k < limit and st not in (cp_model.OPTIMAL, cp_model.INFEASIBLE):
        return -1
    return cb.k
