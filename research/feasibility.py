"""Search linear constructions over Z_b^2 for both variants and verify them."""
import itertools, sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from gamekhelega.games.sudoku3d.rules import Rules, is_complete_solution

def indep(a, b, p):
    return (a[0] * b[1] - a[1] * b[0]) % p != 0

for b in (2, 3):
    vecs = [v for v in itertools.product(range(b), repeat=2) if v != (0, 0)]
    for variant in ("layer", "symmetric"):
        pairs = [(0, 1), (2, 3), (4, 5), (1, 3)]
        if variant == "symmetric":
            pairs += [(1, 5), (3, 5)]
        found = 0
        first = None
        for cols in itertools.product(vecs, repeat=6):
            if all(indep(cols[i], cols[j], b) for i, j in pairs):
                found += 1
                first = first or cols
        n = b * b
        # verify the first one by brute rule check
        r = Rules(n, variant)
        C = np.array(first).T
        z, y, x = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")
        d = np.stack([x // b, x % b, y // b, y % b, z // b, z % b])
        v = np.tensordot(C, d, axes=1) % b
        g = v[0] * b + v[1] + 1
        print(f"b={b} n={n} {variant:9s}: {found} linear constructions; first={first} valid={is_complete_solution(g, r)}")
