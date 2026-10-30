"""Benchmark: solver backends on generated puzzles + empirical clue minimisation."""
import json, sys, time, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from gamekhelega.games.sudoku3d.rules import Rules
from gamekhelega.games.sudoku3d.generator import random_full_grid, minimize_clues, rate
from gamekhelega.games.sudoku3d.solver_numba import solve
from gamekhelega.games.sudoku3d.solver_sat import SatSolver
from research.solver_cpsat import cpsat_count

which = sys.argv[1]
out = {}
if which == "minclues":
    for n, v, runs in [(4, "layer", 200), (4, "symmetric", 200), (9, "layer", 8), (9, "symmetric", 8)]:
        r = Rules(n, v); res = []
        for k in range(runs):
            rng = np.random.default_rng(k)
            t = time.time()
            sol = random_full_grid(r, rng)
            p = minimize_clues(sol, r, rng, time_budget=None)
            dt = time.time() - t
            rt = rate(p, r)
            res.append((rt.clues, dt, rt.nodes))
        cl = [a for a, _, _ in res]
        print(json.dumps({"n": n, "variant": v, "runs": runs, "min_clues": min(cl), "median": float(np.median(cl)),
                          "max": max(cl), "mean_time_s": round(float(np.mean([b for _, b, _ in res])), 3),
                          "median_nodes": float(np.median([c for _, _, c in res]))}), flush=True)
elif which == "solvers":
    for v in ("layer", "symmetric"):
        r = Rules(9, v)
        puzzles = []
        for k in range(5):
            rng = np.random.default_rng(100 + k)
            sol = random_full_grid(r, rng)
            puzzles.append(minimize_clues(sol, r, rng, time_budget=None))
        solve(puzzles[0], r)  # jit warm
        t = time.time(); c1 = [solve(p, r, limit=2).count for p in puzzles]; tn = (time.time() - t) / 5
        t = time.time(); sat = SatSolver(r); tbuild = time.time() - t
        t = time.time(); c2 = [sat.count(p, 2)[0] for p in puzzles]; ts = (time.time() - t) / 5
        t = time.time(); c3 = [cpsat_count(p, r, 2) for p in puzzles]; tc = (time.time() - t) / 5
        print(json.dumps({"variant": v, "clues": [int((p > 0).sum()) for p in puzzles],
                          "numba_s": round(tn, 4), "pysat_build_s": round(tbuild, 3), "pysat_s": round(ts, 4),
                          "cpsat_s_incl_model_build": round(tc, 4), "counts": [c1, c2, c3]}), flush=True)
