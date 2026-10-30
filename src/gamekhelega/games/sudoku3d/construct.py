"""Closed-form base grids + validity-preserving random symmetry transforms."""
from __future__ import annotations

import numpy as np

from .rules import Rules, is_complete_solution

# Coordinates are written base-b: x = b*x1 + x0. value-1 = b*v1 + v0 where
# (v1, v0) = C @ (x1, x0, y1, y0, z1, z0) over Z_b (b prime: 2 or 3).
# Valid iff the column pairs below are linearly independent (see
# research/feasibility.py, which enumerates all such C and verifies them).
_COLS = {
    # Only needs: (x1,x0),(y1,y0),(z1,z0),(x0,y0) independent.
    "layer": ((1, 0), (0, 1), (0, 1), (1, 0), (1, 0), (0, 1)),  # z0 == x0 -> NOT symmetric
    # Also (x0,z0),(y0,z0) independent -> every slice is a sudoku.
    "symmetric": ((1, 0), (0, 1), (0, 1), (1, 0), (1, 0), (1, 1)),
}


def base_grid(rules: Rules) -> np.ndarray:
    """Deterministic valid solution grid (int8, shape (n,n,n), values 1..n)."""
    b, n = rules.b, rules.n
    if b not in (2, 3):
        raise NotImplementedError("linear construction needs prime b (n=4 or 9)")
    cols = np.array(_COLS[rules.variant]).T  # (2, 6)
    z, y, x = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    digits = np.stack([x // b, x % b, y // b, y % b, z // b, z % b])  # (6, n,n,n)
    v = np.tensordot(cols, digits, axes=1) % b  # (2, n,n,n)
    g = (v[0] * b + v[1] + 1).astype(np.int8)
    assert is_complete_solution(g, rules)
    return g


def random_transform(grid: np.ndarray, rules: Rules, rng: np.random.Generator) -> np.ndarray:
    """Apply a random rule-preserving symmetry.

    Always valid: digit relabel; per-axis band permutation + within-band
    permutation (bands of size b) on x, y, z independently.
    Axis permutation: any for ``symmetric``; for ``layer`` only swapping x<->y
    (z is special because boxes live in z-layers).
    """
    n, b = rules.n, rules.b
    g = np.asarray(grid)

    def axis_perm() -> np.ndarray:
        bands = rng.permutation(b)
        return np.concatenate([band * b + rng.permutation(b) for band in bands])

    g = g[axis_perm()][:, axis_perm()][:, :, axis_perm()]
    if rules.variant == "symmetric":
        g = np.transpose(g, rng.permutation(3))
    elif rng.random() < 0.5:
        g = np.transpose(g, (0, 2, 1))
    relabel = np.concatenate([[0], rng.permutation(n) + 1]).astype(np.int8)
    return np.ascontiguousarray(relabel[g])
