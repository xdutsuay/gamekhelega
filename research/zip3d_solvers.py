"""DFS (numba, pruned) vs CP-SAT (AddCircuit) for counting zip3d solutions up to 2.

Random backbite paths with K evenly spaced waypoints. usage: python research/zip3d_solvers.py
Output: research/zip3d_solvers.log.
"""
import time

import numpy as np

from gamekhelega.games.zip3d.generator import puzzle_from_positions, random_hamiltonian_path
from gamekhelega.games.zip3d.solver_cpsat import cpsat_count
from gamekhelega.games.zip3d.solver_dfs import dfs_solve

rng = np.random.default_rng(5)
dfs_solve(puzzle_from_positions(3, random_hamiltonian_path(3, rng), [0, 26]), 2, 1000)  # JIT warm-up
for n, K in [(3, 6), (3, 3), (4, 16), (4, 10), (4, 6), (5, 40), (5, 20), (6, 60)]:
    N = n ** 3
    path = random_hamiltonian_path(n, rng)
    p = puzzle_from_positions(n, path, np.linspace(0, N - 1, K).round().astype(int).tolist())
    t = time.monotonic(); r = dfs_solve(p, 2, 2_000_000); d = time.monotonic() - t
    t = time.monotonic(); k, _ = cpsat_count(p, 2, time_limit=60); c = time.monotonic() - t
    print(f"n={n} K={K}: dfs count={r.count} aborted={r.aborted} nodes={r.nodes} {d:.2f}s | "
          f"cpsat count={k} {c:.2f}s", flush=True)
