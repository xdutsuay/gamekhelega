"""Can randomized search produce full 9^3 grids? (a) from empty with restarts, (b) re-solve a
transformed base grid after blanking a fraction of cells (gives non-linear grids)."""
import time, sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from gamekhelega.games.sudoku3d.rules import Rules, is_complete_solution
from gamekhelega.games.sudoku3d.construct import base_grid, random_transform
from gamekhelega.games.sudoku3d.solver_numba import solve
for v in ("layer", "symmetric"):
    r = Rules(9, v); z = np.zeros(r.shape, np.int8)
    ok = 0; t = time.time()
    for k in range(20):
        s = solve(z, r, limit=1, seed=k, max_nodes=3000); ok += s.count
    print(f"(a) {v}: empty-restarts success {ok}/20 @3000 nodes, {time.time()-t:.1f}s total", flush=True)
    rng = np.random.default_rng(0)
    for frac in (0.5, 0.7, 0.85):
        t = time.time(); succ = diff = 0
        for k in range(10):
            g = random_transform(base_grid(r), r, rng)
            p = np.array(g, order="C"); p.reshape(-1)[rng.random(r.ncells) < frac] = 0
            s = solve(p, r, limit=1, seed=k, max_nodes=3000)
            if s.count:
                succ += 1; assert is_complete_solution(s.solution, r)
                diff += int((s.solution != g).sum() > 0)
        print(f"(b) {v} blank {frac}: solved {succ}/10, differs-from-base {diff}/10, {(time.time()-t)/10:.3f}s each", flush=True)
