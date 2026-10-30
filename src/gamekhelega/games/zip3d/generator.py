"""3D Zip puzzle generation.

1. Random Hamiltonian path: a 3D serpentine, then many *backbite* moves (Mansfield 1982):
   pick an end, connect it to a random grid neighbour already on the path, and reverse
   the tail so the path stays Hamiltonian with a new end.
2. Waypoints: start + end + a few evenly spaced positions along the path.
3. Repair to uniqueness: while a 2nd solution exists, add a waypoint at the first
   step where the alternative diverges from our path (or, with walls on, sometimes a
   wall on the edge the alternative uses but ours does not).
4. Greedy reduction: drop interior waypoints in random order while still unique,
   down to the difficulty's target density (hard = to a local minimum).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .model import ZipConfig, ZipPuzzle, cell, edge, neighbours
from .solver_dfs import dfs_solve

# Hybrid uniqueness: DFS up to this many nodes (~10-30 ms), then CP-SAT. Measured on the box
# (research/zip3d_solvers.log): DFS settles most 3^3 checks; on 4^3/5^3 uniqueness proofs it
# often needs >2M nodes while CP-SAT takes ~0.2 s on 4^3, 1.6-17 s on 5^3, ~43 s on 6^3.
UNIQUE_NODE_CAP = 20_000
DFS_NODES_PER_S = 1_000_000  # fallback budget when OR-Tools is missing
RATE_NODE_CAP = 1_000_000
CHECK_TIME_S = 15.0  # cap on one CP-SAT uniqueness check inside the generator
# Target waypoint count as a fraction of cells (greedy stops once at/below it). Greedy minimal
# puzzles land at ~25-40% of cells (research/zip3d_gen.log), so "hard" = locally minimal.
TARGET_FRAC = {"easy": 0.45, "medium": 0.30, "hard": 0.0}


def serpentine(n: int) -> list[int]:
    path = []
    for z in range(n):
        ys = range(n) if z % 2 == 0 else range(n - 1, -1, -1)
        for i, y in enumerate(ys):
            xs = range(n) if (i + z * n) % 2 == 0 else range(n - 1, -1, -1)
            path.extend(cell(n, x, y, z) for x in xs)
    return path


def random_hamiltonian_path(n: int, rng: np.random.Generator, moves: int | None = None) -> list[int]:
    path = serpentine(n)
    N = len(path)
    if N < 3:
        return path
    nb = [[d for _, d in neighbours(n, c)] for c in range(N)]
    moves = moves if moves is not None else 20 * N * max(1, int(math.log(N)))
    pos = {c: i for i, c in enumerate(path)}
    for _ in range(moves):
        if rng.random() < 0.5:
            path.reverse()
            pos = {c: i for i, c in enumerate(path)}
        endc = path[-1]
        d = nb[endc][int(rng.integers(len(nb[endc])))]
        i = pos[d]
        if i == N - 2:
            continue
        tail = path[i + 1:][::-1]
        path[i + 1:] = tail
        for j, c in enumerate(tail, start=i + 1):
            pos[c] = j
    return path


def puzzle_from_positions(n: int, path: list[int], positions, walls=()) -> ZipPuzzle:
    ps = sorted(set(positions) | {0, len(path) - 1})
    return ZipPuzzle(n, {path[i]: k + 1 for k, i in enumerate(ps)}, set(walls))


def count(p: ZipPuzzle, limit: int = 2, node_cap: int = UNIQUE_NODE_CAP,
          time_limit: float | None = None):
    """Hybrid counter: (count | None, solutions). DFS first, CP-SAT on abort."""
    r = dfs_solve(p, limit=limit, max_nodes=node_cap)
    if not r.aborted:
        return r.count, r.solutions
    from .solver_cpsat import available, cpsat_count
    if not available():
        cap = int(DFS_NODES_PER_S * (time_limit if time_limit is not None else 30))
        r = dfs_solve(p, limit=limit, max_nodes=cap)
        return (None if r.aborted else r.count), r.solutions
    return cpsat_count(p, limit, time_limit=time_limit)


def generate(cfg: ZipConfig, rng: np.random.Generator, difficulty: str = "medium",
             time_budget: float | None = None) -> tuple[ZipPuzzle, list[int]]:
    n, N = cfg.n, cfg.ncells
    t0 = time.monotonic()

    def left():  # per-check CP-SAT limit; an undecided check just keeps the extra clue/wall
        if not time_budget:
            return CHECK_TIME_S
        return min(CHECK_TIME_S, max(0.2, time_budget - (time.monotonic() - t0)))

    path = random_hamiltonian_path(n, rng)
    at = {c: i for i, c in enumerate(path)}
    on_path = {edge(a, b) for a, b in zip(path, path[1:])}
    m = max(2, N // 8)
    positions = set(np.linspace(0, N - 1, m).round().astype(int).tolist())
    walls: set[tuple[int, int]] = set()

    # repair to uniqueness
    while True:
        p = puzzle_from_positions(n, path, positions, walls)
        k, sols = count(p, 2, time_limit=left())
        if k == 1:
            break
        alt = next((s for s in sols if s != path), None)
        if alt is None:  # undecided: add a waypoint in the middle of the largest gap
            ps = sorted(positions | {0, N - 1})
            gaps = [(b - a, a, b) for a, b in zip(ps, ps[1:])]
            _, a, b = max(gaps)
            positions.add((a + b) // 2)
            continue
        i = next(i for i, (u, v) in enumerate(zip(path, alt)) if u != v)
        alt_edge = edge(alt[i - 1], alt[i])
        if cfg.walls and len(walls) < cfg.wall_cap() and alt_edge not in on_path and rng.random() < 0.5:
            walls.add(alt_edge)
        else:
            positions.add(i)
            # the alternative may revisit our cell later; pin a little ahead too
            j = at[alt[i]]
            if 0 < j < N - 1 and rng.random() < 0.3:
                positions.add(j)

    # greedy reduction
    target = int(TARGET_FRAC.get(difficulty, 0.10) * N)
    interior = [i for i in sorted(positions) if 0 < i < N - 1]
    for i in rng.permutation(interior):
        if len(positions | {0, N - 1}) <= max(2, target):
            break
        if time_budget and time.monotonic() - t0 > time_budget:
            break
        trial = positions - {int(i)}
        k, _ = count(puzzle_from_positions(n, path, trial, walls), 2, time_limit=left())
        if k == 1:
            positions = trial
    if walls:
        for w in [sorted(walls)[int(j)] for j in rng.permutation(len(walls))]:
            if time_budget and time.monotonic() - t0 > time_budget:
                break
            trial = walls - {w}
            k, _ = count(puzzle_from_positions(n, path, positions, trial), 2, time_limit=left())
            if k == 1:
                walls = trial
    return puzzle_from_positions(n, path, positions, walls), path


@dataclass
class Rating:
    score: float
    waypoints: int
    nodes: int


def rate(p: ZipPuzzle) -> Rating:
    """0..10: 4*(1 - K/N) + 6*min(1, log10(1+nodes)/5), nodes = DFS nodes to solve + prove unique
    (deterministic neighbour order, capped at RATE_NODE_CAP)."""
    N = p.n ** 3
    r = dfs_solve(p, limit=2, max_nodes=RATE_NODE_CAP)
    nodes = min(r.nodes, RATE_NODE_CAP)
    score = 4 * (1 - p.K / N) + 6 * min(1.0, math.log10(1 + nodes) / 5)
    return Rating(round(score, 2), p.K, nodes)
