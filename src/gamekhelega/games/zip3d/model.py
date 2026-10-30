"""3D Zip data model: grid geometry, puzzles (waypoints + walls), JSON helpers.

Cells are indexed ``c = x + n*y + n*n*z`` with 0-based x, y, z. JSON and prompts are 1-based.
A wall blocks the face between two face-adjacent cells; it is stored as the sorted pair
``(c, c')`` and written in JSON as ``[x, y, z, "+x"|"+y"|"+z"]`` (the wall on the + face of x,y,z).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
        "+z": (0, 0, 1), "-z": (0, 0, -1)}
DIR_NAMES = list(DIRS)


@dataclass
class ZipConfig:
    n: int = 4
    walls: bool = False      # generator may add walls (puzzles from makers may always carry walls)
    max_walls: int = -1      # cap on generated walls; -1 = n

    @property
    def ncells(self) -> int:
        return self.n ** 3

    def wall_cap(self) -> int:
        return self.n if self.max_walls < 0 else self.max_walls

    def to_dict(self) -> dict:
        return {"n": self.n, "walls": self.walls, "max_walls": self.max_walls}


@dataclass
class ZipPuzzle:
    n: int
    waypoints: dict[int, int]                       # cell -> k (1..K, contiguous)
    walls: set[tuple[int, int]] = field(default_factory=set)  # sorted adjacent cell pairs

    @property
    def K(self) -> int:
        return max(self.waypoints.values()) if self.waypoints else 0

    def cell_of(self, k: int) -> int:
        for c, kk in self.waypoints.items():
            if kk == k:
                return c
        raise KeyError(k)


def cell(n: int, x: int, y: int, z: int) -> int:
    return x + n * y + n * n * z


def xyz(n: int, c: int) -> tuple[int, int, int]:
    return c % n, (c // n) % n, c // (n * n)


def neighbours(n: int, c: int):
    """Face-adjacent cells of c, in DIR_NAMES order; yields (dir_name, cell)."""
    x, y, z = xyz(n, c)
    for name, (dx, dy, dz) in DIRS.items():
        a, b, d = x + dx, y + dy, z + dz
        if 0 <= a < n and 0 <= b < n and 0 <= d < n:
            yield name, cell(n, a, b, d)


def adjacent(n: int, a: int, b: int) -> bool:
    ax, ay, az = xyz(n, a)
    bx, by, bz = xyz(n, b)
    return abs(ax - bx) + abs(ay - by) + abs(az - bz) == 1


def edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def colour(n: int, c: int) -> int:
    """Bipartite colour (x+y+z) mod 2: every step flips it."""
    x, y, z = xyz(n, c)
    return (x + y + z) & 1


def neighbour_table(n: int, walls=()) -> np.ndarray:
    """int32 [n^3, 6]: neighbour cell per direction, -1 if off-grid or walled."""
    N = n ** 3
    walls = set(walls)
    t = np.full((N, 6), -1, dtype=np.int32)
    for c in range(N):
        for name, d in neighbours(n, c):
            if edge(c, d) not in walls:
                t[c, DIR_NAMES.index(name)] = d
    return t


def wall_to_json(n: int, w: tuple[int, int]) -> list:
    a, b = w  # a < b, so b is the + neighbour of a
    ax, ay, az = xyz(n, a)
    bx, by, bz = xyz(n, b)
    d = "+x" if bx != ax else ("+y" if by != ay else "+z")
    return [ax + 1, ay + 1, az + 1, d]


def wall_from_json(n: int, w) -> tuple[int, int]:
    if not isinstance(w, (list, tuple)) or len(w) != 4:
        raise ValueError(f"wall {w!r}: expected [x, y, z, \"+x\"|\"+y\"|\"+z\"]")
    x, y, z = (_int(v, "wall coordinate") - 1 for v in w[:3])
    d = str(w[3]).strip().lower()
    if d not in ("+x", "+y", "+z", "-x", "-y", "-z"):
        raise ValueError(f"wall {w!r}: direction must be +x, +y or +z")
    dx, dy, dz = DIRS[d]
    for v in (x, y, z, x + dx, y + dy, z + dz):
        if not 0 <= v < n:
            raise ValueError(f"wall {w!r} is outside the {n}x{n}x{n} grid")
    return edge(cell(n, x, y, z), cell(n, x + dx, y + dy, z + dz))


def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"{what} {v!r} is not an integer")
    try:
        f = float(v)
    except ValueError:
        raise ValueError(f"{what} {v!r} is not an integer") from None
    if f != int(f):
        raise ValueError(f"{what} {v!r} is not an integer")
    return int(f)


def coord_from_json(n: int, p) -> int:
    if not isinstance(p, (list, tuple)) or len(p) != 3:
        raise ValueError(f"cell {p!r}: expected [x, y, z]")
    x, y, z = (_int(v, "coordinate") for v in p)
    if not all(1 <= v <= n for v in (x, y, z)):
        raise ValueError(f"cell {p!r}: coordinates must be 1..{n}")
    return cell(n, x - 1, y - 1, z - 1)


def coord_to_json(n: int, c: int) -> list[int]:
    x, y, z = xyz(n, c)
    return [x + 1, y + 1, z + 1]
