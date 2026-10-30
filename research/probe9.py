"""Instrumented greedy clue removal on one 9^3 grid: progress + node stats per check."""
import sys, time, numpy as np
sys.path.insert(0, "src")
from gamekhelega.games.sudoku3d.rules import Rules
from gamekhelega.games.sudoku3d.generator import random_full_grid
from gamekhelega.games.sudoku3d.solver_numba import solve
variant, seed, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = Rules(9, variant); rng = np.random.default_rng(seed)
sol = random_full_grid(r, rng); p = np.array(sol, order="C"); f = p.reshape(-1)
solve(p, r)
t0 = time.time(); clues = 729; aborted = 0; maxnodes = 0
for i, c in enumerate(rng.permutation(729)):
    v = f[c]; f[c] = 0
    s = solve(p, r, limit=2, max_nodes=cap)
    maxnodes = max(maxnodes, s.nodes)
    if s.count == 1 and not s.aborted: clues -= 1
    else:
        f[c] = v; aborted += s.aborted
    if i % 100 == 99 or i == 728:
        print(f"{variant} seed={seed} tried={i+1} clues={clues} aborted={aborted} max_nodes={maxnodes} t={time.time()-t0:.1f}s", flush=True)
