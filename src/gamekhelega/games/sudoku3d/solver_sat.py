"""Alternative solver: CNF + incremental SAT (python-sat / CaDiCaL).

Backend glucose4 (supports interrupt for time limits; CaDiCaL in pysat does not).
One-hot vars x[c, v]; each unit and each cell gets an exactly-one constraint
(pairwise AMO + ALO clause). Clues are passed as *assumptions*, so one solver
instance is reused across all uniqueness checks of a generation run (learned
clauses persist). Uniqueness: find a model, then re-solve with a guarded
blocking clause (activation literal) to look for a second one.
"""
from __future__ import annotations

import itertools
import threading

import numpy as np
from pysat.solvers import Solver

from .rules import Rules


class SatSolver:
    def __init__(self, rules: Rules, backend: str = "glucose4"):
        self.rules = rules
        n, N = rules.n, rules.ncells
        self.var = lambda c, v: c * n + v + 1  # v in 0..n-1
        self.top = self.nvars = N * n
        self.s = Solver(name=backend)
        groups = [[self.var(c, v) for v in range(n)] for c in range(N)]
        for row in rules.units:
            for v in range(n):
                groups.append([self.var(int(c), v) for c in row])
        for g in groups:
            self.s.add_clause(g)
            for a, b in itertools.combinations(g, 2):
                self.s.add_clause([-a, -b])

    def _assumptions(self, puzzle: np.ndarray) -> list[int]:
        g = np.asarray(puzzle).ravel()
        return [self.var(int(c), int(g[c]) - 1) for c in np.flatnonzero(g)]

    def count(self, puzzle: np.ndarray, limit: int = 2,
              time_limit: float | None = None) -> tuple[int | None, np.ndarray | None]:
        """Returns (count capped at limit, first solution). count None = timed out."""
        n = self.rules.n
        timer = None
        if time_limit:
            timer = threading.Timer(time_limit, self.s.interrupt)
            timer.start()
        try:
            return self._count(puzzle, limit, n, time_limit is not None)
        finally:
            if timer:
                timer.cancel()
                self.s.clear_interrupt()

    def _count(self, puzzle, limit, n, limited):
        assum = self._assumptions(puzzle)
        found, first, act = 0, None, None
        while found < limit:
            extra = [act] if act else []
            res = (self.s.solve_limited(assumptions=assum + extra, expect_interrupt=True)
                   if limited else self.s.solve(assumptions=assum + extra))
            if res is None:
                return None, first
            if not res:
                break
            model = np.array(self.s.get_model()[: self.nvars]) > 0
            sol = (model.reshape(-1, n).argmax(1) + 1).astype(np.int8)
            if first is None:
                first = sol.reshape(self.rules.shape)
            found += 1
            # guarded blocking clause: (not act) or differs somewhere
            self.top += 1
            act = self.top
            pos = np.flatnonzero(model) + 1
            self.s.add_clause([-act] + [-int(p) for p in pos])
        return found, first
