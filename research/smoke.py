import time, sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from gamekhelega.games.sudoku3d.rules import Rules, is_complete_solution
from gamekhelega.games.sudoku3d.solver_numba import solve
t=time.time(); solve(np.zeros((4,4,4),np.int8), Rules(4), limit=1); print("jit %.1fs"%(time.time()-t), flush=True)
for n in (4, 9):
    for v in ("layer", "symmetric"):
        r = Rules(n, v); z = np.zeros(r.shape, np.int8)
        for k in range(3):
            t = time.time(); s = solve(z, r, limit=1, seed=k, max_nodes=200000)
            print(n, v, k, "%.3fs" % (time.time() - t), s.count, s.nodes, s.aborted,
                  s.solution is not None and is_complete_solution(s.solution, r), flush=True)
