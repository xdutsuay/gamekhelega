"""Greedy clue removal with hybrid uniqueness: numba (node cap) -> incremental pysat fallback.
Logs how often fallback is needed and per-backend time. Usage: probe_hybrid.py VARIANT SEED CAP"""
import sys, time, json, numpy as np
sys.path.insert(0, "src")
from gamekhelega.games.sudoku3d.rules import Rules
from gamekhelega.games.sudoku3d.generator import random_full_grid
from gamekhelega.games.sudoku3d.solver_numba import solve
from gamekhelega.games.sudoku3d.solver_sat import SatSolver
variant, seed, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = Rules(9, variant); rng = np.random.default_rng(seed)
sol = random_full_grid(r, rng); p = np.array(sol, order="C"); f = p.reshape(-1)
solve(p, r); t = time.time(); sat = SatSolver(r); tb = time.time() - t
t0 = time.time(); clues = 729; fb = 0; t_nb = t_sat = 0.0; sat_max = 0.0
for c in rng.permutation(729):
    v = f[c]; f[c] = 0
    a = time.time(); s = solve(p, r, limit=2, max_nodes=cap); t_nb += time.time() - a
    if s.aborted:
        fb += 1; a = time.time(); k, _ = sat.count(p, 2); d = time.time() - a; t_sat += d; sat_max = max(sat_max, d)
    else:
        k = s.count
    if k == 1: clues -= 1
    else: f[c] = v
print(json.dumps({"variant": variant, "seed": seed, "cap": cap, "final_clues": clues, "sat_fallbacks": fb,
                  "numba_s": round(t_nb, 2), "sat_s": round(t_sat, 2), "sat_max_s": round(sat_max, 3),
                  "sat_build_s": round(tb, 2), "total_s": round(time.time() - t0, 2)}), flush=True)
