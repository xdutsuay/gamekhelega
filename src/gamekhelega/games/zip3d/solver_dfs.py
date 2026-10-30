"""Exact backtracking solver / solution counter for 3D Zip (numba).

Search: extend the path from waypoint 1, one face-adjacent unvisited cell at a time.
Pruning at every node (all exact, never cuts a real solution):
  * waypoint order: a waypoint cell can only be entered when it is the next number;
    the last waypoint only as the very last cell;
  * degree: every unvisited cell other than the end needs >= 2 usable neighbours
    (unvisited or the path head), the end cell >= 1 (checked around the move only,
    since nothing else changes);
  * connectivity: flood fill from the head over unvisited cells must reach all of them;
  * waypoint reachability: the next waypoint must be in that same component (implied);
  * parity (global, before search): the grid graph is bipartite, so a path of N cells
    from start s to end e needs colour(s) == colour(e) iff N is odd, and the colour
    counts must match.
`limit` caps the number of solutions counted; `max_nodes` aborts (result undecided).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .model import ZipPuzzle, colour, neighbour_table


@njit(cache=True)
def _connected(nbr, visited, head, remaining, queue, mark, stamp):
    """Cells reachable from head through unvisited cells == remaining (unvisited count)?"""
    qh = 0
    qt = 0
    queue[qt] = head
    qt += 1
    mark[head] = stamp
    seen = 0
    while qh < qt:
        c = queue[qh]
        qh += 1
        for j in range(6):
            d = nbr[c, j]
            if d >= 0 and not visited[d] and mark[d] != stamp:
                mark[d] = stamp
                queue[qt] = d
                qt += 1
                seen += 1
    return seen == remaining


@njit(cache=True)
def _next_reachable(nbr, visited, wp, head, target, nextk, queue, mark, stamp):
    """Can the head reach waypoint `target` (number nextk) through unvisited cells
    without stepping on any other waypoint?"""
    qh = 0
    qt = 0
    queue[qt] = head
    qt += 1
    mark[head] = stamp
    while qh < qt:
        c = queue[qh]
        qh += 1
        for j in range(6):
            d = nbr[c, j]
            if d >= 0 and not visited[d] and mark[d] != stamp:
                if d == target:
                    return True
                if wp[d] > 0:
                    continue
                mark[d] = stamp
                queue[qt] = d
                qt += 1
    return False


@njit(cache=True)
def _degree_ok(nbr, visited, cell, head, end):
    """Unvisited `cell` still has enough usable neighbours."""
    cnt = 0
    for j in range(6):
        d = nbr[cell, j]
        if d >= 0 and (not visited[d] or d == head):
            cnt += 1
    if cell == end:
        return cnt >= 1
    return cnt >= 2


@njit(cache=True)
def _search(nbr, wp, wcell, start, end, limit, max_nodes, out):
    N = nbr.shape[0]
    K = wcell.shape[0] - 1
    visited = np.zeros(N, dtype=np.bool_)
    path = np.empty(N, dtype=np.int64)
    nexti = np.zeros(N, dtype=np.int64)
    nk = np.zeros(N, dtype=np.int64)
    queue = np.empty(N, dtype=np.int64)
    mark = np.zeros(N, dtype=np.int64)
    stamp = 0
    count = 0
    nodes = 0
    path[0] = start
    visited[start] = True
    nk[0] = 2
    depth = 0
    if N == 1:
        out[0, 0] = start
        return 1, 0, False
    while depth >= 0:
        if depth == N - 1:
            if count < out.shape[0]:
                for i in range(N):
                    out[count, i] = path[i]
            count += 1
            if count >= limit:
                return count, nodes, False
            visited[path[depth]] = False
            depth -= 1
            continue
        c = path[depth]
        moved = False
        while nexti[depth] < 6:
            j = nexti[depth]
            nexti[depth] += 1
            d = nbr[c, j]
            if d < 0 or visited[d]:
                continue
            k = wp[d]
            if k > 0 and k != nk[depth]:
                continue
            if d == end and depth + 1 != N - 1:
                continue
            nodes += 1
            if nodes > max_nodes:
                return count, nodes, True
            visited[d] = True
            ok = True
            remaining = N - depth - 2
            if remaining > 0:
                # degree check on cells whose usable-neighbour count just changed
                for jj in range(6):
                    e = nbr[c, jj]
                    if e >= 0 and not visited[e] and not _degree_ok(nbr, visited, e, d, end):
                        ok = False
                        break
                if ok:
                    for jj in range(6):
                        e = nbr[d, jj]
                        if e >= 0 and not visited[e] and not _degree_ok(nbr, visited, e, d, end):
                            ok = False
                            break
                if ok:
                    stamp += 1
                    ok = _connected(nbr, visited, d, remaining, queue, mark, stamp)
                if ok and k == 0:
                    nxt = nk[depth]
                    if nxt <= K:
                        stamp += 1
                        ok = _next_reachable(nbr, visited, wp, d, wcell[nxt], nxt, queue, mark, stamp)
            if ok:
                depth += 1
                path[depth] = d
                nexti[depth] = 0
                nk[depth] = nk[depth - 1] + (1 if k > 0 else 0)
                moved = True
                break
            visited[d] = False
        if not moved:
            visited[path[depth]] = False
            depth -= 1
    return count, nodes, False


@dataclass
class DfsResult:
    count: int
    solutions: list[list[int]]
    nodes: int
    aborted: bool


def parity_ok(p: ZipPuzzle) -> bool:
    n, N = p.n, p.n ** 3
    if p.K < 2 or N < 2:
        return N == 1
    s, e = p.cell_of(1), p.cell_of(p.K)
    black = sum(colour(n, c) for c in range(N))
    white = N - black
    if N % 2 == 0:
        return black == white and colour(n, s) != colour(n, e)
    major = 1 if black > white else 0
    return abs(black - white) == 1 and colour(n, s) == major and colour(n, e) == major


def dfs_solve(p: ZipPuzzle, limit: int = 2, max_nodes: int = 2_000_000,
              rng: np.random.Generator | None = None) -> DfsResult:
    """Count solutions (capped at limit). rng shuffles neighbour order (random solutions)."""
    n, N = p.n, p.n ** 3
    if p.K < 2 or not parity_ok(p):
        return DfsResult(0, [], 0, False)
    nbr = neighbour_table(n, p.walls)
    if rng is not None:
        for c in range(N):
            nbr[c] = nbr[c, rng.permutation(6)]
    wp = np.zeros(N, dtype=np.int64)
    for c, k in p.waypoints.items():
        wp[c] = k
    wcell = np.full(p.K + 1, -1, dtype=np.int64)
    for c, k in p.waypoints.items():
        wcell[k] = c
    out = np.full((max(1, limit), N), -1, dtype=np.int64)
    cnt, nodes, aborted = _search(nbr, wp, wcell, p.cell_of(1), p.cell_of(p.K), limit, max_nodes, out)
    sols = [out[i].tolist() for i in range(min(cnt, out.shape[0]))]
    return DfsResult(int(cnt), sols, int(nodes), bool(aborted))
