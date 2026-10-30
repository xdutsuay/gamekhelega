"""3D Zip: LinkedIn's Zip in a cube. STUB with pseudocode only - not implemented.

Rules: n x n x n grid (or a x b x c). Some cells carry waypoint numbers 1..K.
Draw ONE path of orthogonally adjacent steps (6-neighbourhood) that
  * starts on waypoint 1 and ends on waypoint K,
  * visits waypoints in increasing order,
  * visits every cell exactly once (Hamiltonian path),
  * never crosses a wall (a wall blocks the face between two adjacent cells).

Data model
  Config:   {"n": 4, "walls": true, "max_waypoints": 12}
  Puzzle:   waypoints: dict[(x,y,z) -> k] (k = 1..K, contiguous),
            walls: set[frozenset{(x,y,z),(x',y',z')}] (unordered adjacent pairs)
  JSON:     {"waypoints": [[x,y,z,k], ...], "walls": [[x,y,z,"+x"|"+y"|"+z"], ...]}  (1-based)
  Solution: ordered list of n^3 cells; JSON {"path": [[x,y,z], ...]}

check_solution (cheap, exact):
  len(path) == n^3 and set(path) == all cells
  every consecutive pair is 6-adjacent and not walled
  path[0] is waypoint 1, path[-1] is waypoint K
  positions of waypoints along path strictly increasing in k
  partial (with reference) = longest common prefix with reference / n^3

count_solutions(limit=2) - SAT/CP-SAT (pysat reuse like sudoku3d.solver_sat):
  var pos[c][t] (cell c at step t), exactly-one per c and per t      [O(N^2) vars; ok for n<=4]
  adjacency: pos[c][t] -> OR(pos[d][t+1] for d in open_neighbours(c))
  waypoint k at cell c with fixed known index? no: order only ->
      for k<k': pos[c_k][t] -> OR(pos[c_k'][t'] for t' > t)  (or ordinal vars)
  pos[c_1][0] = 1, pos[c_K][N-1] = 1
  Alternative (smaller, better for n>=5): successor encoding succ[c][d] for open
  edges + exactly-one in/out + subtour elimination via CP-SAT AddCircuit with a
  dummy node and waypoint order via integer rank vars (rank[c_k] < rank[c_k+1]).
  -> Prefer CP-SAT AddCircuit; enumerate up to 2 solutions with a guarded
     blocking clause / solution callback StopSearch.

generate(difficulty, rng):
  1. Random Hamiltonian path on the grid graph: start from a serpentine
     (boustrophedon) path through the cube, then apply many "backbite" moves
     (pick an endpoint, pick a random neighbour d not its path-neighbour, add
     edge, remove the edge that closes the cycle, new endpoint) -> near-uniform.
  2. Waypoints: always path[0] (k=1) and path[-1] (k=K); add a few more along
     the path at random spaced positions.
  3. Loop: count_solutions(limit=2). If 2: take the second solution, find the
     first index where it diverges from ours, and either add a waypoint at our
     path cell near there, or (if walls on) add a wall on the edge the other
     solution uses but ours does not. Repeat until unique.
  4. Then try removing waypoints/walls one by one (random order) keeping
     uniqueness (same greedy as sudoku3d.generator.minimize_clues).
  difficulty: fewer waypoints/walls = harder; rate by solver conflicts/time.

prompts: describe coordinates (1-based x,y,z), 6-neighbour moves, walls notation,
  and output {"path": [[x,y,z],...]} of length n^3.
render_terminal: one panel per z-layer, waypoint numbers bold, path drawn with
  step indices, walls as thick separators, layer-change marks "↑/↓" on cells
  where the path goes to z+1/z-1.
render_3d: plotly Scatter3d line through path (color by step), waypoint labels,
  walls as small quads.
human: "move DIR" (+x,-x,+y,-y,+z,-z) from the path head, "undo", "goto x y z".
"""
from __future__ import annotations

from ...engine.game import PuzzleGame

_TODO = "zip3d is a stub; see module docstring for the design"


class Zip3D(PuzzleGame):
    name = "zip3d"
    title = "3D Zip (stub)"
    description = "One path through numbered waypoints filling an n^3 cube. NOT IMPLEMENTED."

    def config_schema(self):
        return {"type": "object", "properties": {
            "n": {"type": "integer", "minimum": 3, "maximum": 6, "default": 4},
            "walls": {"type": "boolean", "default": True},
            "max_waypoints": {"type": "integer", "default": 12}}}

    def parse_config(self, d):
        return {"n": 4, "walls": True, "max_waypoints": 12, **(d or {})}

    def config_to_json(self, cfg):
        return dict(cfg)

    def generate(self, cfg, difficulty, rng, time_budget=None):
        raise NotImplementedError(_TODO)

    def validate_puzzle(self, cfg, puzzle):
        raise NotImplementedError(_TODO)

    def check_solution(self, cfg, puzzle, solution, reference=None):
        raise NotImplementedError(_TODO)

    def count_solutions(self, cfg, puzzle, limit=2, time_limit=None):
        raise NotImplementedError(_TODO)

    def solve(self, cfg, puzzle, time_limit=None):
        raise NotImplementedError(_TODO)

    def puzzle_to_json(self, cfg, puzzle):
        raise NotImplementedError(_TODO)

    def puzzle_from_json(self, cfg, data):
        raise NotImplementedError(_TODO)

    def solution_to_json(self, cfg, solution):
        raise NotImplementedError(_TODO)

    def solution_from_json(self, cfg, data):
        raise NotImplementedError(_TODO)

    def prompt_rules(self, cfg):
        raise NotImplementedError(_TODO)

    def puzzle_json_schema(self, cfg):
        return {"type": "object", "required": ["waypoints"], "properties": {
            "waypoints": {"type": "array", "items": {"type": "array", "minItems": 4, "maxItems": 4}},
            "walls": {"type": "array", "items": {"type": "array", "minItems": 4, "maxItems": 4}}}}

    def solution_json_schema(self, cfg):
        return {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "array", "items": {"type": "array", "minItems": 3, "maxItems": 3}}}}


GAME = Zip3D()
