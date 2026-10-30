"""Minimal template plugin: "latin" = n x n Latin square (every row/column holds 1..n once).

Copy this file to src/gamekhelega/games/<name>/__init__.py (built-in) or ship it in
your own package with an entry point:

    [project.entry-points."gamekhelega.games"]
    latin = "my_pkg.latin:GAME"

Brute-force solver: fine for n <= 5 only. Real games should use SAT/numba.
"""
from __future__ import annotations

import itertools

import numpy as np

from gamekhelega.engine.game import CheckResult, PuzzleGame


class Latin(PuzzleGame):
    name, title = "latin", "Latin square (template)"
    description = "n x n grid, each row and column holds 1..n once."

    def config_schema(self):
        return {"type": "object", "properties": {"n": {"type": "integer", "minimum": 3, "maximum": 5}}}

    def parse_config(self, d):
        return {"n": int((d or {}).get("n", 4))}

    def config_to_json(self, cfg):
        return dict(cfg)

    def _ok(self, g):
        n = g.shape[0]
        s = np.arange(1, n + 1)
        return all((np.sort(g, 0) == s[:, None]).all(0)) and all((np.sort(g, 1) == s[None, :]).all(1))

    def _solutions(self, cfg, p, limit):
        n, out = cfg["n"], []
        rows = [list(r) for r in itertools.permutations(range(1, n + 1))]

        def rec(i, g):
            if len(out) >= limit:
                return
            if i == n:
                out.append(np.array(g)); return
            for r in rows:
                if all(p[i][j] in (0, r[j]) for j in range(n)) and \
                        all(r[j] != g[k][j] for k in range(i) for j in range(n)):
                    rec(i + 1, g + [r])
        rec(0, [])
        return out

    def generate(self, cfg, difficulty, rng, time_budget=None):
        n = cfg["n"]
        sol = (np.add.outer(np.arange(n), np.arange(n)) % n + 1)[rng.permutation(n)][:, rng.permutation(n)]
        p = sol.copy()
        for c in rng.permutation(n * n):
            v = p.flat[c]; p.flat[c] = 0
            if len(self._solutions(cfg, p, 2)) != 1:
                p.flat[c] = v
        return p, sol

    def validate_puzzle(self, cfg, p):
        return [] if p.shape == (cfg["n"],) * 2 else ["bad shape"]

    def check_solution(self, cfg, p, s, reference=None):
        ok = s.shape == p.shape and bool(((p == 0) | (p == s)).all()) and self._ok(s)
        partial = float((s == reference).mean()) if reference is not None and not ok else float(ok)
        return CheckResult(ok, [] if ok else ["invalid"], partial)

    def count_solutions(self, cfg, p, limit=2, time_limit=None):
        return len(self._solutions(cfg, p, limit))

    def solve(self, cfg, p, time_limit=None):
        s = self._solutions(cfg, p, 1)
        return s[0] if s else None

    def puzzle_to_json(self, cfg, p):
        return {"rows": p.tolist()}

    def puzzle_from_json(self, cfg, d):
        return np.array(d["rows"], dtype=int)

    solution_to_json, solution_from_json = puzzle_to_json, puzzle_from_json

    def prompt_rules(self, cfg):
        return (f'Latin square {cfg["n"]}x{cfg["n"]}: each row and column holds 1..{cfg["n"]} once. '
                'JSON {"rows": [[...], ...]}, 0 = empty.')

    def puzzle_json_schema(self, cfg):
        return {"type": "object", "required": ["rows"]}

    solution_json_schema = puzzle_json_schema

    def render_terminal(self, cfg, p, solution=None, state=None):
        g = solution if solution is not None else p
        return "\n".join(" ".join(str(v or ".") for v in row) for row in g)


GAME = Latin()
