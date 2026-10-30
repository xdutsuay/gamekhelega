"""3D Zip as a PuzzleGame plugin (registry key "zip3d").

Rules (LinkedIn's Zip in a cube): an n x n x n grid with numbered waypoints 1..K.
Draw ONE path through face-adjacent cells (6-neighbourhood) that visits every cell
exactly once, starts on 1, ends on K, passes the waypoints in increasing order and
never crosses a wall. A duel puzzle must have exactly one such path.

Puzzle JSON: {"waypoints": [[x, y, z, k], ...], "walls": [[x, y, z, "+x"|"+y"|"+z"], ...]} (1-based;
             a wall [x,y,z,"+x"] blocks the face between (x,y,z) and (x+1,y,z)).
Solution JSON: {"path": [[x, y, z], ...]} with all n^3 cells in order; for convenience
             {"start": [x, y, z], "moves": "+x +y -z ..."} is also accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from ...engine.game import CheckResult, Move, PuzzleGame
from . import generator as gen
from .model import (DIRS, ZipConfig, ZipPuzzle, adjacent, cell, coord_from_json, coord_to_json,
                    edge, wall_from_json, wall_to_json, xyz, _int)
from .solver_dfs import dfs_solve


@dataclass
class PlayState:
    path: list[int]                     # cells so far, starts on waypoint 1
    walls: set = field(default_factory=set)
    layer: int | None = None            # z-layer shown (None = all)
    history: list = field(default_factory=list)  # previous paths (undo)


class Zip3D(PuzzleGame):
    name = "zip3d"
    title = "3D Zip"
    description = ("n x n x n cube with numbered waypoints; draw one face-adjacent path through every "
                   "cell, from 1 to K in order, never crossing a wall.")

    # ---- config -------------------------------------------------------
    def config_schema(self):
        return {"type": "object", "properties": {
            "n": {"type": "integer", "minimum": 2, "maximum": 6, "default": 4},
            "walls": {"type": "boolean", "default": False,
                      "description": "let the generator add walls (submitted puzzles may always have walls)"},
            "max_walls": {"type": "integer", "default": -1, "description": "generator wall cap; -1 = n"}},
            "additionalProperties": False}

    def parse_config(self, d: dict | None) -> ZipConfig:
        d = dict(d or {})
        extra = set(d) - {"n", "walls", "max_walls"}
        if extra:
            raise ValueError(f"unknown zip3d config keys: {sorted(extra)}")
        cfg = ZipConfig(n=int(d.get("n", 4)), walls=bool(d.get("walls", False)),
                        max_walls=int(d.get("max_walls", -1)))
        if not 2 <= cfg.n <= 6:
            raise ValueError("zip3d n must be 2..6")
        return cfg

    def config_to_json(self, cfg: ZipConfig) -> dict:
        return cfg.to_dict()

    # ---- core ----------------------------------------------------------
    def generate(self, cfg, difficulty, rng, time_budget=None):
        return gen.generate(cfg, rng, difficulty, time_budget=time_budget)

    def validate_puzzle(self, cfg, puzzle: ZipPuzzle) -> list[str]:
        errs = []
        if puzzle.n != cfg.n:
            errs.append(f"puzzle is {puzzle.n}^3, config says {cfg.n}^3")
        ks = sorted(puzzle.waypoints.values())
        if len(ks) < 2:
            errs.append("need at least 2 waypoints (1 and K)")
        elif ks != list(range(1, len(ks) + 1)):
            errs.append(f"waypoint numbers must be 1..K with no gaps or repeats, got {ks}")
        return errs

    def check_solution(self, cfg, puzzle: ZipPuzzle, solution: list[int], reference=None) -> CheckResult:
        n, N = cfg.n, cfg.ncells
        path = list(solution)
        errs = []
        if len(path) != N:
            errs.append(f"path has {len(path)} cells, need {N}")
        if len(set(path)) != len(path):
            errs.append(f"{len(path) - len(set(path))} cells visited more than once")
        bad = [i for i, (a, b) in enumerate(zip(path, path[1:])) if not adjacent(n, a, b)]
        if bad:
            errs.append(f"{len(bad)} steps are not between face-adjacent cells, first at step {bad[0] + 1}")
        walled = [i for i, (a, b) in enumerate(zip(path, path[1:])) if edge(a, b) in puzzle.walls]
        if walled:
            errs.append(f"{len(walled)} steps cross a wall, first at step {walled[0] + 1}")
        if path and puzzle.K >= 2:
            if path[0] != puzzle.cell_of(1):
                errs.append("path does not start on waypoint 1")
            if path[-1] != puzzle.cell_of(puzzle.K):
                errs.append(f"path does not end on waypoint {puzzle.K}")
            seen = [puzzle.waypoints[c] for c in path if c in puzzle.waypoints]
            if seen != sorted(seen) or len(seen) != puzzle.K:
                errs.append("waypoints are not visited in order 1..K")
        ok = not errs
        partial = 1.0 if ok else 0.0
        if not ok and reference is not None and N > 1:
            ref = {edge(a, b) for a, b in zip(reference, reference[1:])}
            mine = {edge(a, b) for a, b in zip(path, path[1:]) if adjacent(n, a, b)}
            partial = len(ref & mine) / (N - 1)
        return CheckResult(ok, errs, partial)

    def count_solutions(self, cfg, puzzle, limit=2, time_limit=None):
        k, _ = gen.count(puzzle, limit, time_limit=time_limit)
        return k

    def solve(self, cfg, puzzle, time_limit=None):
        r = dfs_solve(puzzle, limit=1, max_nodes=gen.UNIQUE_NODE_CAP * 10)
        if r.count:
            return r.solutions[0]
        if not r.aborted:
            return None
        k, sols = gen.count(puzzle, 1, time_limit=time_limit)
        return sols[0] if sols else None

    def rate(self, cfg, puzzle):
        return gen.rate(puzzle).score

    # ---- serialization -------------------------------------------------
    def puzzle_to_json(self, cfg, puzzle: ZipPuzzle):
        n = cfg.n
        wps = sorted(puzzle.waypoints.items(), key=lambda t: t[1])
        return {"waypoints": [coord_to_json(n, c) + [k] for c, k in wps],
                "walls": [wall_to_json(n, w) for w in sorted(puzzle.walls)]}

    def puzzle_from_json(self, cfg, data) -> ZipPuzzle:
        n = cfg.n
        if not isinstance(data, dict) or not isinstance(data.get("waypoints"), list):
            raise ValueError('expected {"waypoints": [[x,y,z,k], ...], "walls": [...]}')
        wps: dict[int, int] = {}
        for w in data["waypoints"]:
            if not isinstance(w, (list, tuple)) or len(w) != 4:
                raise ValueError(f"waypoint {w!r}: expected [x, y, z, k]")
            c = coord_from_json(n, w[:3])
            k = _int(w[3], "waypoint number")
            if c in wps:
                raise ValueError(f"two waypoints on cell {w[:3]}")
            wps[c] = k
        walls = data.get("walls") or []
        if not isinstance(walls, list):
            raise ValueError("walls must be a list")
        return ZipPuzzle(n, wps, {wall_from_json(n, w) for w in walls})

    def solution_to_json(self, cfg, solution):
        return {"path": [coord_to_json(cfg.n, c) for c in solution]}

    def solution_from_json(self, cfg, data) -> list[int]:
        n = cfg.n
        if isinstance(data, dict) and "path" not in data and "moves" in data:
            c = coord_from_json(n, data.get("start"))
            path = [c]
            for mv in str(data["moves"]).replace(",", " ").split():
                path.append(_step(n, path[-1], mv))
            return path
        p = data.get("path") if isinstance(data, dict) else data
        if not isinstance(p, list):
            raise ValueError('expected {"path": [[x,y,z], ...]}')
        return [coord_from_json(n, q) for q in p]

    def puzzle_json_schema(self, cfg):
        n = cfg.n
        coord = {"type": "integer", "minimum": 1, "maximum": n}
        return {"type": "object", "required": ["waypoints"], "properties": {
            "waypoints": {"type": "array", "minItems": 2, "items": {
                "type": "array", "prefixItems": [coord, coord, coord, {"type": "integer", "minimum": 1}],
                "minItems": 4, "maxItems": 4}},
            "walls": {"type": "array", "items": {
                "type": "array", "prefixItems": [coord, coord, coord, {"enum": ["+x", "+y", "+z"]}],
                "minItems": 4, "maxItems": 4}}}}

    def solution_json_schema(self, cfg):
        n = cfg.n
        coord = {"type": "integer", "minimum": 1, "maximum": n}
        return {"type": "object", "required": ["path"], "properties": {"path": {
            "type": "array", "minItems": n ** 3, "maxItems": n ** 3,
            "items": {"type": "array", "items": coord, "minItems": 3, "maxItems": 3}}}}

    # ---- prompts -------------------------------------------------------
    def prompt_rules(self, cfg):
        n, N = cfg.n, cfg.ncells
        return (
            f"3D Zip, grid {n}x{n}x{n} ({N} cells), coordinates x, y, z each 1..{n}.\n"
            "1. Some cells carry waypoint numbers 1..K (consecutive).\n"
            "2. Draw ONE path that visits EVERY cell exactly once. Each step moves to a face-adjacent cell: "
            "change exactly one of x, y, z by 1 (6 directions +x -x +y -y +z -z, no diagonals).\n"
            "3. The path starts on waypoint 1, ends on waypoint K, and meets the waypoints in increasing "
            "order (1, 2, 3, ..., K).\n"
            '4. Walls: [x, y, z, "+x"] blocks the step between (x,y,z) and (x+1,y,z); likewise "+y", "+z". '
            "The path may never cross a wall.\n"
            f'PUZZLE FORMAT: {{"waypoints": [[x, y, z, k], ...], "walls": [[x, y, z, "+x"|"+y"|"+z"], ...]}}.\n'
            f'SOLUTION FORMAT: {{"path": [[x, y, z], ...]}} listing all {N} cells in visiting order '
            "(first = waypoint 1, last = waypoint K).\n"
            'Example (2x2x2): waypoints [[1,1,1,1],[1,2,1,2]] -> path [[1,1,1],[2,1,1],[2,1,2],[1,1,2],'
            "[1,2,2],[2,2,2],[2,2,1],[1,2,1]].")

    def prompt_make_tips(self, cfg):
        n = cfg.n
        par = ("the grid has an even number of cells, so waypoint 1 and K must have different colour "
               "(x+y+z odd vs even)" if n % 2 == 0 else
               "the grid has an odd number of cells, so waypoint 1 and K must both have x+y+z even")
        return (f"Build a path first: a serpentine through each layer (alternate the row direction, then step "
                f"+z and come back) is valid; scramble it with 'backbite' moves (link an end to a neighbouring "
                f"path cell, reverse the tail). Then place waypoints along it: 1 on the first cell, K on the "
                f"last, and enough in between that NO other path fits; walls on edges your path does not use "
                f"also rule out alternatives. Parity: {par}. Fewer waypoints is harder, but a puzzle with 2+ "
                "solutions costs you heavily.")

    def prompt_solve_tips(self, cfg):
        return ("Colour cells by (x+y+z) mod 2; every step flips the colour. Corner and edge cells have few "
                "neighbours, so the path is often forced there. Never cut the unvisited cells into two parts. "
                "Check that your path has every cell exactly once before answering.")

    # ---- rendering -----------------------------------------------------
    def render_terminal(self, cfg, puzzle, solution=None, state: PlayState | None = None):
        from .render import render_terminal
        path = state.path if state is not None else solution
        return render_terminal(puzzle, path, None if state is None else state.layer)

    def render_3d(self, cfg, puzzle, solution=None):
        from .render import render_3d
        return render_3d(puzzle, solution)

    # ---- human ---------------------------------------------------------
    def human_help(self, cfg):
        n = cfg.n
        return (f"coords are 1..{n}. The path starts on waypoint 1 (blue = path head). commands:\n"
                "  +x -y +z ...  (or: move +x -y)   step the head; several moves at once are fine\n"
                "  goto X Y Z                      step to an adjacent cell, or cut the path back to a cell on it\n"
                "  undo | reset | view [K]         view K = show only z-layer K, view = all layers\n"
                "  check                           count visited cells and rule problems (no hints)")

    def new_state(self, cfg, puzzle):
        return PlayState(path=[puzzle.cell_of(1)], walls=set(puzzle.walls))

    def parse_move(self, cfg, text: str) -> Move:
        t = text.lower().replace(",", " ").split()
        if not t:
            raise ValueError("empty command")
        if t[0] in DIRS:
            t = ["move"] + t
        cmd, args = t[0], t[1:]
        if cmd == "move":
            if not args or any(a not in DIRS for a in args):
                raise ValueError("usage: move +x|-x|+y|-y|+z|-z ...")
            return Move("move", {"dirs": args})
        if cmd == "goto":
            if len(args) != 3 or not all(a.isdigit() and 1 <= int(a) <= cfg.n for a in args):
                raise ValueError(f"usage: goto X Y Z (1..{cfg.n})")
            return Move("goto", {"x": int(args[0]), "y": int(args[1]), "z": int(args[2])})
        if cmd == "view":
            k = None
            if args:
                if not args[-1].isdigit() or not 1 <= int(args[-1]) <= cfg.n:
                    raise ValueError(f"usage: view [K] with K in 1..{cfg.n}")
                k = int(args[-1])
            return Move("view", {"k": k})
        if cmd in ("undo", "reset", "check"):
            return Move(cmd)
        raise ValueError(f"unknown command {cmd!r}; type help")

    def apply_move(self, cfg, state: PlayState, move: Move) -> str:
        n = cfg.n
        a = move.args
        if move.kind == "move":
            new = list(state.path)
            for d in a["dirs"]:
                try:
                    c = _step(n, new[-1], d)
                except ValueError as e:
                    return str(e)
                if c in new:
                    return f"{d}: cell {coord_to_json(n, c)} already on the path"
                if edge(new[-1], c) in state.walls:
                    return f"{d}: a wall blocks that step"
                new.append(c)
            state.history.append(state.path)
            state.path = new
            return f"ok ({len(new)}/{cfg.ncells})"
        if move.kind == "goto":
            c = cell(n, a["x"] - 1, a["y"] - 1, a["z"] - 1)
            if c in state.path:
                state.history.append(state.path)
                state.path = state.path[: state.path.index(c) + 1]
                return f"cut back to step {len(state.path)}"
            if not adjacent(n, state.path[-1], c):
                return "not adjacent to the path head"
            if edge(state.path[-1], c) in state.walls:
                return "a wall blocks that step"
            state.history.append(state.path)
            state.path = state.path + [c]
            return f"ok ({len(state.path)}/{cfg.ncells})"
        if move.kind == "view":
            state.layer = None if a["k"] is None else a["k"] - 1
            return "viewing all layers" if a["k"] is None else f"viewing z={a['k']}"
        if move.kind == "undo":
            if not state.history:
                return "nothing to undo"
            state.path = state.history.pop()
            return "undone"
        if move.kind == "reset":
            state.history.append(state.path)
            state.path = state.path[:1]
            return "reset to waypoint 1"
        if move.kind == "check":
            return f"{len(state.path)}/{cfg.ncells} cells visited"
        raise ValueError(move.kind)

    def state_to_solution(self, cfg, state: PlayState):
        return list(state.path)


def _step(n: int, c: int, d: str) -> int:
    if d not in DIRS:
        raise ValueError(f"bad move {d!r}; use +x -x +y -y +z -z")
    x, y, z = xyz(n, c)
    dx, dy, dz = DIRS[d]
    x, y, z = x + dx, y + dy, z + dz
    if not all(0 <= v < n for v in (x, y, z)):
        raise ValueError(f"move {d} leaves the grid")
    return cell(n, x, y, z)


GAME = Zip3D()
