"""Solver/uniqueness benchmark on 9^3 puzzles produced by the real generator.
usage: python research/bench_solvers.py <variant> <difficulty> <k_puzzles> <gen_budget_s>"""
import json, sys, time, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from gamekhelega.games.sudoku3d.rules import Rules
from gamekhelega.games.sudoku3d.generator import generate
from gamekhelega.games.sudoku3d.solver_numba import solve
from gamekhelega.games.sudoku3d.solver_sat import SatSolver
from research.solver_cpsat import cpsat_count

v, diff, k, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
NUMBA_CAP = 100_000  # 2M took 379 s and still aborted on a 39-clue symmetric puzzle
r = Rules(9, v)
solve(np.zeros(r.shape, np.int8), r, limit=1, max_nodes=10)  # jit warm-up (capped: empty-cube DFS does not finish for layer)
t = time.time(); sat = SatSolver(r); tb = time.time() - t
for i in range(k):
    t = time.time(); p, s = generate(r, np.random.default_rng(500 + i), diff, time_budget=budget)
    tg = time.time() - t
    t = time.time(); rn = solve(p, r, limit=2, max_nodes=NUMBA_CAP); tn = time.time() - t
    t = time.time(); cs, _ = sat.count(p, 2, time_limit=120); ts = time.time() - t
    t = time.time(); cc = cpsat_count(p, r, 2, time_limit=120); tc = time.time() - t
    print(json.dumps({"numba_cap": NUMBA_CAP, "variant": v, "difficulty": diff, "i": i, "clues": int((p > 0).sum()), "gen_s": round(tg, 2),
                      "numba": {"count": rn.count, "aborted": bool(rn.aborted), "nodes": int(rn.nodes), "s": round(tn, 4)},
                      "pysat_glucose4": {"count": cs, "s": round(ts, 4), "build_s": round(tb, 3)},
                      "cpsat_1worker_incl_build": {"count": cc, "s": round(tc, 3)}}), flush=True)
