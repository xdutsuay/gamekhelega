"""Sudoku3D as a PuzzleGame plugin."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...engine.game import CheckResult, Move, PuzzleGame
from . import generator as gen
from .rules import VARIANTS, Rules, conflicts, is_complete_solution
from .solver_numba import solve as nb_solve

ALPHABET = "123456789ABCDEFG"  # value v -> ALPHABET[v-1]; "0" or "." = empty
AXES = {"z": 0, "y": 1, "x": 2}  # axis name -> numpy axis of g[z, y, x]


def _sat(rules: Rules):
    from .solver_sat import SatSolver
    return SatSolver(rules)


@dataclass
class PlayState:
    grid: np.ndarray                  # current entries, givens included
    givens: np.ndarray                # bool mask
    axis: str = "z"                   # slice axis shown
    layer: int | None = None          # None = show all slices along axis
    history: list = field(default_factory=list)  # (flat_idx, old_value)


class Sudoku3D(PuzzleGame):
    name = "sudoku3d"
    title = "3D Sudoku"
    description = ("n x n x n cube; every x/y/z line holds 1..n once; boxes in every z-layer "
                   "(variant 'layer') or in every slice of all three axes (variant 'symmetric').")

    # ---- config -------------------------------------------------------
    def config_schema(self) -> dict:
        return {"type": "object", "properties": {
            "n": {"type": "integer", "enum": [4, 9], "default": 9},
            "variant": {"type": "string", "enum": list(VARIANTS), "default": "layer"}},
            "additionalProperties": False}

    def parse_config(self, d: dict | None) -> Rules:
        d = dict(d or {})
        extra = set(d) - {"n", "variant"}
        if extra:
            raise ValueError(f"unknown sudoku3d config keys: {sorted(extra)}")
        return Rules.from_dict(d)

    def config_to_json(self, cfg: Rules) -> dict:
        return cfg.to_dict()

    # ---- core ----------------------------------------------------------
    def generate(self, cfg: Rules, difficulty: str, rng: np.random.Generator,
                 time_budget: float | None = None):
        return gen.generate(cfg, rng, difficulty, time_budget=time_budget)

    def validate_puzzle(self, cfg: Rules, puzzle: np.ndarray) -> list[str]:
        if puzzle.shape != cfg.shape:
            return [f"shape {puzzle.shape} != {cfg.shape}"]
        if puzzle.min() < 0 or puzzle.max() > cfg.n:
            return ["values out of range"]
        bad = conflicts(puzzle, cfg)
        if bad.any():
            z, y, x = np.argwhere(bad)[0]
            return [f"{int(bad.sum())} givens conflict, e.g. at x={x+1} y={y+1} z={z+1}"]
        return []

    def check_solution(self, cfg: Rules, puzzle: np.ndarray, solution: np.ndarray,
                       reference: np.ndarray | None = None) -> CheckResult:
        errs = []
        if solution.shape != cfg.shape:
            return CheckResult(False, [f"shape {solution.shape} != {cfg.shape}"])
        given = puzzle > 0
        if (solution[given] != puzzle[given]).any():
            errs.append(f"{int((solution[given] != puzzle[given]).sum())} givens changed")
        if (solution == 0).any():
            errs.append(f"{int((solution == 0).sum())} cells empty")
        bad = conflicts(solution, cfg)
        if bad.any():
            errs.append(f"{int(bad.sum())} cells in rule conflicts")
        ok = not errs and is_complete_solution(solution, cfg)
        partial = 1.0 if ok else 0.0
        if not ok and reference is not None:
            free = ~given
            partial = float((solution[free] == reference[free]).mean()) if free.any() else 0.0
        return CheckResult(ok, errs, partial)

    def count_solutions(self, cfg: Rules, puzzle: np.ndarray, limit: int = 2,
                        time_limit: float | None = None) -> int | None:
        t0 = time.monotonic()
        r = nb_solve(puzzle, cfg, limit=limit, max_nodes=gen.UNIQUE_NODE_CAP)
        if not r.aborted:
            return r.count
        left = None if time_limit is None else max(0.5, time_limit - (time.monotonic() - t0))
        k, _ = _sat(cfg).count(puzzle, limit, time_limit=left)
        return k

    def solve(self, cfg: Rules, puzzle: np.ndarray, time_limit: float | None = None):
        r = nb_solve(puzzle, cfg, limit=1, max_nodes=gen.UNIQUE_NODE_CAP * 4)
        if r.count:
            return r.solution
        if not r.aborted:
            return None
        _, sol = _sat(cfg).count(puzzle, 1, time_limit=time_limit)
        return sol

    def rate(self, cfg: Rules, puzzle: np.ndarray) -> float:
        return gen.rate(puzzle, cfg).score

    # ---- serialization -------------------------------------------------
    def _to_layers(self, g: np.ndarray) -> list[list[str]]:
        lut = np.array(list("0" + ALPHABET))
        return [["".join(lut[row]) for row in layer] for layer in np.asarray(g)]

    def _from_layers(self, cfg: Rules, data: Any, allow_empty: bool) -> np.ndarray:
        layers = data.get("layers") if isinstance(data, dict) else data
        n = cfg.n
        if not isinstance(layers, list) or len(layers) != n:
            raise ValueError(f'expected {{"layers": [{n} layers]}}')
        g = np.zeros(cfg.shape, dtype=np.int8)
        for z, layer in enumerate(layers):
            if not isinstance(layer, list) or len(layer) != n:
                raise ValueError(f"layer {z+1}: expected {n} row strings")
            for y, row in enumerate(layer):
                row = str(row).replace(" ", "").upper()
                if len(row) != n:
                    raise ValueError(f"layer {z+1} row {y+1}: expected {n} chars, got {len(row)}")
                for x, ch in enumerate(row):
                    if ch in "0.":
                        if not allow_empty:
                            raise ValueError(f"empty cell at x={x+1} y={y+1} z={z+1} in a solution")
                        continue
                    v = ALPHABET.find(ch) + 1
                    if not 1 <= v <= n:
                        raise ValueError(f"bad char {ch!r} at x={x+1} y={y+1} z={z+1}")
                    g[z, y, x] = v
        return g

    def puzzle_to_json(self, cfg, puzzle):
        return {"layers": self._to_layers(puzzle)}

    def puzzle_from_json(self, cfg, data):
        return self._from_layers(cfg, data, allow_empty=True)

    def solution_to_json(self, cfg, solution):
        return {"layers": self._to_layers(solution)}

    def solution_from_json(self, cfg, data):
        return self._from_layers(cfg, data, allow_empty=False)

    def _grid_schema(self, cfg: Rules, empty: bool) -> dict:
        chars = ("0" if empty else "") + ALPHABET[: cfg.n]
        row = {"type": "string", "pattern": f"^[{chars}]{{{cfg.n}}}$"}
        return {"type": "object", "required": ["layers"], "properties": {"layers": {
            "type": "array", "minItems": cfg.n, "maxItems": cfg.n,
            "items": {"type": "array", "minItems": cfg.n, "maxItems": cfg.n, "items": row}}}}

    def puzzle_json_schema(self, cfg):
        return self._grid_schema(cfg, True)

    def solution_json_schema(self, cfg):
        return self._grid_schema(cfg, False)

    # ---- prompts -------------------------------------------------------
    def prompt_rules(self, cfg: Rules) -> str:
        n, b = cfg.n, cfg.b
        digits = ALPHABET[:n]
        extra = ("" if cfg.variant == "layer" else
                 f"\n4. Also every x-slice (fixed x) and every y-slice (fixed y), viewed as an {n}x{n} "
                 f"grid, has its {b}x{b} boxes each containing {digits[0]}..{digits[-1]} once "
                 "(so EVERY axis-aligned slice is a valid 2D sudoku).")
        return (
            f"3D Sudoku, size {n}x{n}x{n} ({n**3} cells), coordinates x (column), y (row), z (layer), "
            f"each 1..{n}. Symbols: {digits}.\n"
            f"1. Every line along x (fixed y,z), along y (fixed x,z) and along z (fixed x,y) contains "
            f"each symbol exactly once.\n"
            f"2. Every horizontal layer (fixed z) is a valid {n}x{n} sudoku: its {b}x{b} boxes "
            f"(x and y split into bands of {b}) each contain each symbol exactly once.\n"
            f"3. A puzzle has some cells given; the solution fills all cells and keeps all givens."
            f"{extra}\n"
            f'FORMAT: {{"layers": [layer z=1, ..., layer z={n}]}}; each layer is a list of {n} strings '
            f"(rows y=1..{n}), each string has {n} chars (x=1..{n}). '0' = empty (puzzles only).\n"
            f"Example of one layer row: \"{digits}\" means x=1 holds {digits[0]}, x={n} holds {digits[-1]}."
        )

    def prompt_make_tips(self, cfg):
        b, n = cfg.b, cfg.n
        sym = ("" if cfg.variant == "layer" else
               " (that pattern satisfies the 'layer' rules only; for this variant you need a cube whose x- and "
               "y-slices also have valid boxes, so check all slices)")
        return (f"A valid full cube for the layer rules (1-based symbols): value(x,y,z) = "
                f"((x-1) + {b}*((y-1) mod {b}) + (y-1) div {b} + (z-1)) mod {n} + 1{sym}. "
                "Disguise it: relabel symbols, permute bands and rows/columns inside a band, permute layers. "
                "Then remove givens while the solution stays unique. Fewer givens is usually harder, but "
                "uniqueness is mandatory: a non-unique puzzle costs you heavily.")

    def prompt_solve_tips(self, cfg):
        return ("Use candidate elimination across all three line directions and the layer boxes; "
                "check every line before answering.")

    # ---- rendering -----------------------------------------------------
    def render_terminal(self, cfg, puzzle, solution=None, state: PlayState | None = None):
        from .render import render_terminal
        return render_terminal(cfg, puzzle, solution, state)

    def render_3d(self, cfg, puzzle, solution=None):
        from .render import render_3d
        return render_3d(cfg, puzzle, solution)

    # ---- human ---------------------------------------------------------
    def human_help(self, cfg):
        n = cfg.n
        return (f"coords are 1..{n}. commands:\n"
                f"  set X Y Z V   (or just: X Y Z V)   put V at x=X y=Y z=Z\n"
                f"  clear X Y Z                      empty a cell (givens are locked)\n"
                f"  view AXIS [K]                    AXIS in x|y|z; show slice K or all slices\n"
                f"  undo | check (count conflicts/empties, no hints)")

    def new_state(self, cfg, puzzle):
        return PlayState(grid=np.array(puzzle, dtype=np.int8, order="C"), givens=puzzle > 0)

    def parse_move(self, cfg, text: str) -> Move:
        t = text.lower().split()
        if not t:
            raise ValueError("empty command")
        if t[0].isdigit():
            t = ["set"] + t
        cmd, args = t[0], t[1:]

        def nums(k: int) -> list[int]:
            if len(args) != k or not all(a.isdigit() or a.upper() in ALPHABET for a in args):
                raise ValueError(f"usage: {cmd} needs {k} numbers")
            vals = [int(a) if a.isdigit() else ALPHABET.index(a.upper()) + 1 for a in args]
            if not all(1 <= v <= cfg.n for v in vals):
                raise ValueError(f"numbers must be 1..{cfg.n}")
            return vals

        if cmd == "set":
            x, y, z, v = nums(4)
            return Move("set", {"x": x, "y": y, "z": z, "v": v})
        if cmd == "clear":
            x, y, z = nums(3)
            return Move("clear", {"x": x, "y": y, "z": z})
        if cmd == "view":
            if not args or args[0] not in AXES:
                raise ValueError("usage: view x|y|z [K]")
            k = None
            if len(args) > 1:
                k = int(args[1])
                if not 1 <= k <= cfg.n:
                    raise ValueError(f"K must be 1..{cfg.n}")
            return Move("view", {"axis": args[0], "k": k})
        if cmd in ("undo", "check"):
            return Move(cmd)
        raise ValueError(f"unknown command {cmd!r}; type help")

    def apply_move(self, cfg, state: PlayState, move: Move) -> str:
        a = move.args
        if move.kind in ("set", "clear"):
            z, y, x = a["z"] - 1, a["y"] - 1, a["x"] - 1
            if state.givens[z, y, x]:
                return "that cell is a given (locked)"
            idx = int(np.ravel_multi_index((z, y, x), cfg.shape))
            state.history.append((idx, int(state.grid[z, y, x])))
            state.grid[z, y, x] = a["v"] if move.kind == "set" else 0
            return "ok"
        if move.kind == "view":
            state.axis, state.layer = a["axis"], (None if a["k"] is None else a["k"] - 1)
            return f"viewing {a['axis']}-slices" + ("" if a["k"] is None else f" #{a['k']}")
        if move.kind == "undo":
            if not state.history:
                return "nothing to undo"
            idx, old = state.history.pop()
            state.grid.reshape(-1)[idx] = old
            return "undone"
        if move.kind == "check":
            bad = int(conflicts(state.grid, cfg).sum())
            empty = int((state.grid == 0).sum())
            return f"{empty} empty, {bad} cells in conflict"
        raise ValueError(move.kind)

    def state_to_solution(self, cfg, state: PlayState):
        return state.grid.copy()


GAME = Sudoku3D()
