"""Default solver: numba-jitted bitmask DFS with constraint propagation.

Propagation = naked singles (peer elimination) + hidden singles (per unit).
Branching = MRV (fewest candidates), value order from a caller permutation,
so the same code does solving, uniqueness counting and random grid generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .rules import Rules


@nb.njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


@nb.njit(cache=True)
def _propagate(cand, fixed, peers, units, n):
    """In-place. Returns False on contradiction."""
    N = cand.shape[0]
    changed = True
    while changed:
        changed = False
        for c in range(N):
            m = cand[c]
            if m == 0:
                return False
            if fixed[c] == 0 and (m & (m - 1)) == 0:
                fixed[c] = 1
                changed = True
                for p in peers[c]:
                    if cand[p] & m:
                        cand[p] &= ~m
                        if cand[p] == 0:
                            return False
        # hidden singles
        for u in range(units.shape[0]):
            for v in range(n):
                bit = np.uint32(1) << np.uint32(v)
                cnt = 0
                last = -1
                for c in units[u]:
                    if cand[c] & bit:
                        cnt += 1
                        last = c
                        if cnt > 1:
                            break
                if cnt == 0:
                    return False
                if cnt == 1 and cand[last] != bit:
                    cand[last] = bit
                    changed = True
    return True


@nb.njit(cache=True)
def _mrv(cand, fixed):
    best = -1
    bestc = 99
    for c in range(cand.shape[0]):
        if fixed[c] == 0:
            k = _popcount(cand[c])
            if k < bestc:
                bestc = k
                best = c
                if k == 2:
                    break
    return best


@nb.njit(cache=True)
def _search(cand0, peers, units, n, limit, perm, max_nodes, out):
    """Returns (n_solutions_found<=limit, nodes, max_depth, aborted). First solution -> out."""
    N = cand0.shape[0]
    S = np.empty((N + 2, N), dtype=np.uint32)
    F = np.zeros((N + 2, N), dtype=np.uint8)
    cell = np.empty(N + 2, dtype=np.int64)
    rem = np.zeros(N + 2, dtype=np.uint32)
    S[0] = cand0
    nodes = 0
    maxd = 0
    found = 0
    if not _propagate(S[0], F[0], peers, units, n):
        return 0, 0, 0, False
    c = _mrv(S[0], F[0])
    if c < 0:
        for i in range(N):
            out[i] = 0
            m = S[0][i]
            while m > 1:
                m >>= 1
                out[i] += 1
            out[i] += 1
        return 1, 0, 0, False
    lvl = 0
    cell[0] = c
    rem[0] = S[0][c]
    while lvl >= 0:
        if rem[lvl] == 0:
            lvl -= 1
            continue
        bit = np.uint32(0)
        for v in perm:
            b = np.uint32(1) << np.uint32(v)
            if rem[lvl] & b:
                bit = b
                break
        rem[lvl] &= ~bit
        nodes += 1
        if max_nodes > 0 and nodes > max_nodes:
            return found, nodes, maxd, True
        S[lvl + 1] = S[lvl]
        F[lvl + 1] = F[lvl]
        S[lvl + 1][cell[lvl]] = bit
        if not _propagate(S[lvl + 1], F[lvl + 1], peers, units, n):
            continue
        c = _mrv(S[lvl + 1], F[lvl + 1])
        if c < 0:
            found += 1
            if found == 1:
                for i in range(N):
                    m = S[lvl + 1][i]
                    k = 1
                    while m > 1:
                        m >>= 1
                        k += 1
                    out[i] = k
            if found >= limit:
                return found, nodes, maxd, False
            continue
        lvl += 1
        if lvl > maxd:
            maxd = lvl
        cell[lvl] = c
        rem[lvl] = S[lvl][c]
    return found, nodes, maxd, False


@dataclass
class SolveResult:
    count: int  # solutions found, capped at `limit`
    solution: np.ndarray | None  # first solution (n,n,n) int8
    nodes: int  # branch nodes (0 = solved by pure propagation)
    max_depth: int
    aborted: bool = False  # hit max_nodes before finishing


def _initial_cand(puzzle: np.ndarray, rules: Rules) -> np.ndarray | None:
    full = np.uint32((1 << rules.n) - 1)
    g = np.asarray(puzzle, dtype=np.int64).ravel()
    cand = np.full(g.shape[0], full, dtype=np.uint32)
    mask = g > 0
    cand[mask] = (np.uint32(1) << (g[mask] - 1).astype(np.uint32))
    return cand


def solve(puzzle: np.ndarray, rules: Rules, limit: int = 2, seed: int | None = None,
          max_nodes: int = 0) -> SolveResult:
    """Search for up to `limit` solutions. limit=2 => uniqueness check."""
    g = np.asarray(puzzle)
    if g.shape != rules.shape or g.min() < 0 or g.max() > rules.n:
        raise ValueError("bad puzzle shape/values")
    cand = _initial_cand(g, rules)
    perm = (np.arange(rules.n) if seed is None
            else np.random.default_rng(seed).permutation(rules.n)).astype(np.int64)
    out = np.zeros(rules.ncells, dtype=np.int8)
    cnt, nodes, maxd, aborted = _search(cand, rules.peers, rules.units, rules.n, limit, perm,
                                        max_nodes, out)
    sol = out.reshape(rules.shape) if cnt else None
    return SolveResult(int(cnt), sol, int(nodes), int(maxd), bool(aborted))


def count_solutions(puzzle: np.ndarray, rules: Rules, limit: int = 2) -> int:
    """Uncapped exact count. Only for small or easy puzzles: may run for hours on
    sparse 9^3 grids (use generator.is_unique / plugin.count_solutions instead)."""
    return solve(puzzle, rules, limit=limit).count
