"""Puzzle generation: full grid -> greedy clue removal keeping uniqueness."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .construct import base_grid, random_transform
from .rules import Rules
from .solver_numba import solve

# Hybrid uniqueness: numba DFS up to this many branch nodes, then fall back to
# incremental SAT (research/hybrid.log: pure numba with a 20k cap aborted on dozens
# of checks and stalled ~110 clues on 'layer'; hybrid finished every run).
UNIQUE_NODE_CAP = 500
RATE_NODE_CAP = 10_000  # rating saturates at 10k nodes anyway


def random_full_grid(rules: Rules, rng: np.random.Generator, mix_rounds: int = 3,
                     blank_frac: float = 0.75) -> np.ndarray:
    """Random valid solution grid.

    Pure DFS from an empty 9^3 cube does not finish (research/gen_diversity.log),
    so: transformed closed-form grid, then `mix_rounds` of "blank a random
    fraction and re-solve with a random value order", which can land on
    non-linear grids, then a final random symmetry transform.
    """
    g = random_transform(base_grid(rules), rules, rng)
    for _ in range(mix_rounds):
        p = np.array(g, dtype=np.int8, order="C")
        p.reshape(-1)[rng.random(rules.ncells) < blank_frac] = 0
        s = solve(p, rules, limit=1, seed=int(rng.integers(1 << 31)), max_nodes=2000)
        if s.count:
            g = s.solution
    return random_transform(g, rules, rng)


def is_unique(puzzle: np.ndarray, rules: Rules, node_cap: int = UNIQUE_NODE_CAP,
              sat=None, sat_time_limit: float | None = None) -> bool:
    """numba first; if it hits node_cap, ask the SAT solver (created if not given).
    A SAT timeout counts as 'not unique' (conservative)."""
    r = solve(puzzle, rules, limit=2, max_nodes=node_cap)
    if not r.aborted:
        return r.count == 1
    if sat is None:
        from .solver_sat import SatSolver
        sat = SatSolver(rules)
    k, _ = sat.count(puzzle, 2, time_limit=sat_time_limit)
    return k == 1


def minimize_clues(solution: np.ndarray, rules: Rules, rng: np.random.Generator,
                   target_clues: int = 0, time_budget: float | None = None,
                   node_cap: int = UNIQUE_NODE_CAP) -> np.ndarray:
    """Greedy: visit cells in random order, drop a clue if the puzzle stays unique.

    Stops when clue count <= target_clues, time is up, or every cell was tried.
    Result is *locally* minimal (no single clue removable) if it runs to the end.
    """
    from .solver_sat import SatSolver

    t0 = time.monotonic()
    sat = SatSolver(rules)  # one incremental instance for the whole run
    p = np.array(solution, dtype=np.int8, order="C")  # C order: ravel() must be a view
    flat = p.ravel()
    clues = rules.ncells
    for c in rng.permutation(rules.ncells):
        if clues <= target_clues or (time_budget and time.monotonic() - t0 > time_budget):
            break
        v = flat[c]
        flat[c] = 0
        left = None if not time_budget else max(0.05, time_budget - (time.monotonic() - t0))
        if is_unique(p, rules, node_cap, sat, left):
            clues -= 1
        else:
            flat[c] = v
    return p


@dataclass
class Rating:
    clues: int
    nodes: int  # branch nodes to solve+prove uniqueness (0 = pure propagation)
    max_depth: int
    score: float  # difficulty in [0, 10]


def rate(puzzle: np.ndarray, rules: Rules) -> Rating:
    """Difficulty = blend of emptiness and search effort.

    score = 4*(1 - clues/ncells) + 6*min(1, log10(1 + nodes)/4)
    (nodes from the deterministic uniqueness search; 10k+ nodes saturates, so the
    search is capped at RATE_NODE_CAP and an aborted search counts as the cap).
    """
    clues = int((np.asarray(puzzle) > 0).sum())
    r = solve(puzzle, rules, limit=2, max_nodes=RATE_NODE_CAP)
    score = 4 * (1 - clues / rules.ncells) + 6 * min(1.0, float(np.log10(1 + r.nodes)) / 4)
    return Rating(clues, r.nodes, r.max_depth, round(score, 2))


DIFFICULTY_TARGETS = {  # fraction of cells kept as clues (min reachable ~ see DESIGN.md)
    "easy": 0.45,
    "medium": 0.30,
    "hard": 0.0,  # run greedy to local minimum
}


def generate(rules: Rules, rng: np.random.Generator, difficulty: str = "medium",
             time_budget: float | None = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (puzzle, solution)."""
    sol = random_full_grid(rules, rng)
    target = int(DIFFICULTY_TARGETS[difficulty] * rules.ncells)
    return minimize_clues(sol, rules, rng, target_clues=target, time_budget=time_budget), sol
