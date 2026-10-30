"""Rule sets for 3D sudoku.

Grid convention: numpy int8 array of shape (n, n, n) indexed ``g[z, y, x]``;
``g[z]`` is horizontal layer z. 0 = empty, 1..n = digits. n = b*b (b = box side).

Variants
--------
``layer`` (canonical): every line along x, y and z holds 1..n once (Latin cube)
    and every z-layer's b x b boxes hold 1..n once -> each z-layer is a valid
    2D sudoku.
``symmetric``: additionally every x-slice and every y-slice has valid boxes,
    i.e. *every* axis-aligned slice is a valid 2D sudoku.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

VARIANTS = ("layer", "symmetric")


@dataclass(frozen=True)
class Rules:
    n: int = 9
    variant: str = "layer"

    def __post_init__(self) -> None:
        b = int(round(self.n ** 0.5))
        if b * b != self.n or self.n < 4:
            raise ValueError("n must be a perfect square >= 4 (4, 9, 16)")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")

    @property
    def b(self) -> int:
        return int(round(self.n ** 0.5))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n, self.n, self.n)

    @property
    def ncells(self) -> int:
        return self.n ** 3

    def to_dict(self) -> dict:
        return {"n": self.n, "variant": self.variant}

    @classmethod
    def from_dict(cls, d: dict) -> "Rules":
        return cls(n=int(d.get("n", 9)), variant=d.get("variant", "layer"))

    @cached_property
    def units(self) -> np.ndarray:
        """int32 array (U, n) of flat cell indices; each row must hold 1..n once."""
        n, b = self.n, self.b
        idx = np.arange(n ** 3).reshape(n, n, n)  # [z, y, x]
        units = []
        for i in range(n):
            for j in range(n):
                units.append(idx[i, j, :])  # x-lines
                units.append(idx[i, :, j])  # y-lines
                units.append(idx[:, i, j])  # z-lines

        def boxes(slices):  # slices: list of (n, n) 2D index arrays
            for s in slices:
                for by in range(b):
                    for bx in range(b):
                        units.append(s[by * b:(by + 1) * b, bx * b:(bx + 1) * b].ravel())

        boxes([idx[z] for z in range(n)])  # z-layers (y, x)
        if self.variant == "symmetric":
            boxes([idx[:, y, :] for y in range(n)])  # y-slices (z, x)
            boxes([idx[:, :, x] for x in range(n)])  # x-slices (z, y)
        return np.ascontiguousarray(np.array(units, dtype=np.int32))

    @cached_property
    def cell_units(self) -> np.ndarray:
        """int32 (ncells, k): units containing each cell (k = 4 or 6)."""
        lists = [[] for _ in range(self.ncells)]
        for u, row in enumerate(self.units):
            for c in row:
                lists[c].append(u)
        return np.array(lists, dtype=np.int32)

    @cached_property
    def peers(self) -> np.ndarray:
        """int32 (ncells, P): distinct other cells sharing a unit with each cell."""
        out = []
        for c in range(self.ncells):
            s = set(self.units[self.cell_units[c]].ravel().tolist())
            s.discard(c)
            out.append(sorted(s))
        return np.array(out, dtype=np.int32)


def is_complete_solution(grid: np.ndarray, rules: Rules) -> bool:
    """True iff grid is full and every unit contains 1..n exactly once."""
    g = np.asarray(grid).ravel()
    if g.shape[0] != rules.ncells or g.min() < 1 or g.max() > rules.n:
        return False
    vals = np.sort(g[rules.units], axis=1)
    return bool((vals == np.arange(1, rules.n + 1)).all())


def conflicts(grid: np.ndarray, rules: Rules) -> np.ndarray:
    """Bool mask (n,n,n) of filled cells that clash with a same-value peer."""
    g = np.asarray(grid).ravel()
    bad = np.zeros(g.shape[0], dtype=bool)
    for row in rules.units:
        v = g[row]
        filled = v > 0
        if not filled.any():
            continue
        vals, counts = np.unique(v[filled], return_counts=True)
        dup = vals[counts > 1]
        if dup.size:
            bad[row[np.isin(v, dup)]] = True
    return bad.reshape(rules.shape)
