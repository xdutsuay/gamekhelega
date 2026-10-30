# gamekhelega: design

A game-agnostic referee for **AI-vs-AI (or human) puzzle duels**. Each side
**makes** puzzles for the other and then **solves** the other's puzzles, under
wall-clock limits. The referee checks everything exactly. Games are plugins. The
first plugin is **3D Sudoku** (`sudoku3d`). The second, **3D Zip** (`zip3d`), is
implemented as well (§15).

Status markers: ✅ implemented and tested on the box · 🟡 implemented but rough · ⬜ not done (see CLOUD_AGENT_BRIEF.md)

---

## 1. Layout

```
src/gamekhelega/
  cli.py                     ✅ typer CLI (games/generate/solve/check/play/duel/view/replay/leaderboard)
  engine/                    ✅ ZERO game-specific code (enforced by tests/test_engine.py)
    game.py                  PuzzleGame ABC, CheckResult, Move
    registry.py              built-in games/* modules + entry-point group "gamekhelega.games"
    protocol.py              Request envelope, JSON schemas, exact prompts, lenient JSON extraction
    validate.py              check_submission (rules / solution / uniqueness / rating)
    duel.py                  DuelConfig, run_duel, replay log, hard timeouts
    scoring.py, elo.py       scoring formula, Elo per game
    results.py, viewer.py    rich tables, replay walker, HTML export shell
    io.py                    puzzle pack files, k=v params
    adapters/                mock, openai (compatible HTTP), command (subprocess), human
  games/
    sudoku3d/                ✅ full plugin (rules, construct, numba + SAT solvers, generator, render)
    zip3d/                   ✅ full plugin (model, numba DFS + CP-SAT hybrid, generator, render)
examples/minimal_game.py     ✅ template plugin (Latin square); a mock duel runs on it in tests
research/                    experiments + logs that back the numbers below
configs/                     duel configs (mock 4³, mock 9³, LLM example)
```

## 2. 3D Sudoku rules (canonical: `variant="layer"`)

Cube n×n×n, n = b², b ∈ {2, 3} (so 4³ or 9³). Coordinates x (column), y (row) and z (layer) run 1..n. Symbols are `123456789` (n=9) or `1234` (n=4).

1. **Latin cube**: every line along x, along y and along z contains each symbol exactly once.
2. **Layer boxes**: every z-layer is a valid n×n sudoku (b×b boxes). So each layer is a 2D sudoku, *and* the cube is Latin in z.
3. A puzzle is a partial cube with **exactly one** completion.

No 3×3×3 sub-cubes (those over-constrain the cube). This was decided before the research.

**Variant `symmetric`** (option): it adds b×b boxes to every x-slice and every y-slice, so *every* axis-aligned slice is a sudoku. It is fully supported (`-p variant=symmetric`).

Feasibility, from `research/feasibility.py`: I enumerated linear constructions `value-1 = b·v1 + v0`, where `(v1, v0) = C·(x1, x0, y1, y0, z1, z0) mod b` over GF(b). A construction is valid iff the column pairs (x1,x0), (y1,y0), (z1,z0) and (x0,y0) are independent. The symmetric variant also needs (x0,z0) and (y0,z0).

| | layer | symmetric |
|---|---|---|
| b=2 constructions | 144 | 48 |
| b=3 constructions | 82,944 | 41,472 |

**Why `layer` is canonical:** it is the rule set the owner described ("each layer is a sudoku"). It has fewer constraints, so the same clue count is harder, and it is easier to explain in a prompt. `symmetric` generates faster and needs fewer clues (§5). It is a good "hard-rules / easy-search" option.

Unit counts (each unit must hold 1..n once):
- layer: 3·n² lines + n² layer boxes = 324 units for 9³.
- symmetric: 3·n² lines + 3·n² boxes = 486 units for 9³.

## 3. Data model and JSON

- In memory: `numpy.int8` array `g[z, y, x]`, with 0 = empty. Rules are expressed as `units` (int32 rows of n flat cell indices), `cell_units` and `peers`. The solvers and checkers only use these, so adding a variant just means adding units.
- Puzzle and solution JSON (both canonical and prompt format): `{"layers": [ layer z=1 .. z=n ]}`. Each layer is n strings (rows y=1..n) of n chars (x=1..n). For a puzzle, `0` or `.` means empty. A solution must have no empties.
- Pack file (`generate` output, `check`/`solve`/`view` input):
  ```json
  {"format": "gamekhelega/v1", "game": "sudoku3d", "config": {"n": 9, "variant": "layer"},
   "items": [{"id": "p1", "puzzle": {...}, "solution": {...}, "meta": {"difficulty": 6.1, "gen_time_s": 2.3}}]}
  ```

## 4. Solvers and uniqueness (research/, numbers in §5)

- **numba DFS** (`solver_numba.py`): bitmask candidates, naked and hidden singles over all units, MRV branching, an iterative stack, and seedable value order. `limit=2` counts solutions up to 2. `max_nodes` aborts. It is very fast while propagation does the work. On sparse 9³ *layer* puzzles it can blow up: pure-numba uniqueness with a 20k-node cap aborted 49 times and stalled at 111 clues after 191 s (`research/probe9*.log`).
- **Incremental SAT** (`solver_sat.py`, pysat glucose4): the one-hot CNF (exactly-one per cell / unit / value) is built once per (n, variant), in 0.07 s for 9³. Clues are passed as **assumptions**. "Is there a 2nd solution?" adds a blocking clause guarded by a fresh activation literal, so the instance is reused for every check. It has a time limit (Timer + `interrupt`; CaDiCaL in pysat cannot be interrupted, so it uses glucose4).
- **Hybrid** (the default everywhere): numba up to `UNIQUE_NODE_CAP=500` nodes, then SAT. A SAT timeout counts as "not unique" during generation and as `undecided` during validation.
- **CP-SAT** (`research/solver_cpsat.py`, ortools): one-hot + `AddExactlyOne`, with a solution callback that stops at the limit. For sudoku3d it is benchmarked only; zip3d uses ortools at runtime (§15).

## 5. Empirical results (box: 8 vCPU, Python 3.13, numba 0.68, pysat 1.9.dev17, ortools 9.15)

All clue counts below are **empirical results of a greedy remover** (random order, drop a clue if still unique). They are local minima, **not** proven minimums. For comparison, the 2D sudoku minimum of 17 is a proven result (McGuire et al. 2012). Nothing comparable is known for 3D.

All runs are single-process timings on the box, with up to 3 benchmarks running in parallel. Raw JSON is in `research/*.log`; the scripts are `research/bench_solvers.py`, `probe9.py`, `probe_hybrid.py` and `bench.py`.

**4×4×4 (n=4, b=2): greedy minimum over 200 random full grids** (`research/minclues.log`)

| variant | min clues | median | max | mean time/puzzle |
|---|---|---|---|---|
| layer | 5 | 8 | 11 | 0.002 s |
| symmetric | 4 | 5 | 7 | 0.002 s |

**9×9×9 generation and uniqueness**: generator = greedy remover with hybrid uniqueness (numba DFS capped at 500 nodes, then incremental pysat).

| variant | difficulty | gen budget | clues reached | gen time | numba DFS (cap 100k nodes) | pysat glucose4 (cold) | CP-SAT 1 worker (incl. build) |
|---|---|---|---|---|---|---|---|
| layer | medium (30%) | 90 s | 218 | 0.14-0.15 s | unique, 0 nodes, 0.0004 s | 0.003 s | 0.03 s |
| layer | hard | 90 s | 105 / 109 | 114 s / 101 s | aborted at 100k nodes (19 s) | 91 s / 31 s | 26 s / 10 s |
| layer | hard | 240 s | 105 | 272 s | aborted (20 s) | 90 s | 27 s |
| symmetric | medium (30%) | 90 s | 218 | ≈0.18 s | unique, 0 nodes, 0.0004 s | 0.003-0.005 s | 0.04-0.05 s |
| symmetric | hard | 90 s | 39 / 38 / 39 | 23 s / 77 s / 91 s | aborted at 100k nodes (17-19 s) | 2.2 s / 30 s / 34 s | 2.2 s / 16 s / 16 s |
| symmetric | hard | (cap 2M nodes) | 39 | 22 s | **aborted at 2M nodes after 379 s** | 2.3 s | 2.3 s |

Every completed pysat and CP-SAT run agreed on uniqueness (count = 1).

Other 9³ data points:
- **Symmetric hybrid run to a local minimum** (seed 0, no time budget): 43 clues in 7.3 s total; 44 SAT fallbacks; slowest SAT call 0.48 s (`hybrid.log`).
- **Pure numba, 20k-node cap** (`probe9_*.log`):
  - symmetric: 47 clues in 90 s.
  - layer: stuck at 111 clues after 191 s, with 49 aborted checks.
- **Layer hybrid run to a local minimum did not finish.** Two seeds were killed after about 25 min each without reaching one. With a time budget, layer stalls at about **105-111 clues**: 8 s gives 111, and both 90 s and 240 s give 105 for the same seed.
- The generator overshoots its time budget: the last uniqueness check can run past it. Measured: 114 s for a 90 s budget, 272 s for 240 s.

Empirical 9³ minima (greedy local minima, **not proven**):
- **layer: about 105 clues** (best seen; no local minimum reached);
- **symmetric: 38 clues** (best seen; typically 38-43).

Conclusions:
1. Numba DFS is only fast when propagation alone solves the puzzle (medium and easy). Proving uniqueness of sparse 9³ puzzles needs SAT or CP-SAT.
2. Cold proofs on hard layer puzzles take 30-90 s with pysat and 10-27 s with CP-SAT. The duel referee default is therefore `uniqueness_time_s = 180`.
3. Switching the referee's backend to CP-SAT (an optional `[research]` extra today) would make it about 3× faster on hard layer puzzles. It stays optional so that the base install stays light.
4. Layer is much harder to make sparse than symmetric (≈105 vs ≈38 clues). It is still the canonical variant because those are the owner's rules.

**Difficulty rating** (`generator.rate`, 0-10): `score = 4·(1 − clues/ncells) + 6·min(1, log10(1+nodes)/4)`. Here `nodes` is the number of branch nodes the deterministic numba search needs to solve and prove uniqueness, capped at 10k (the score saturates there). Propagation-only puzzles score ≤ 4. Difficulty presets keep these fractions of cells as clues: easy 45%, medium 30%, hard = greedy to a local minimum (or until the time budget ends).

Random full grids: DFS from an empty 9³ cube did not finish (0/20 within 3000 nodes; 200k nodes took 15 s with no result). So `random_full_grid` = random rule-preserving transform of a linear construction, then 3 rounds of "blank 75% and re-solve with a random value order" (this can move off the linear family; see `research/gen_diversity.log`), then another transform. Transforms used:
- per-axis band permutation and within-band permutation;
- symbol relabelling;
- axis permutations (all 6 for symmetric; only x↔y for layer).

## 6. Plugin interface (`engine/game.py`)

```python
class PuzzleGame(ABC):
    name: str; title: str; description: str
    # config
    def config_schema(self) -> dict                          # JSON Schema of the game config
    def parse_config(self, d: dict) -> Any                   # -> game's config object
    def config_to_json(self, cfg) -> dict
    # core
    def generate(self, cfg, difficulty, rng, time_budget=None) -> (puzzle, solution)
    def validate_puzzle(self, cfg, puzzle) -> list[str]      # structural/rule errors of givens
    def check_solution(self, cfg, puzzle, solution, reference=None) -> CheckResult(ok, errors, partial)
    def count_solutions(self, cfg, puzzle, limit=2, time_limit=None) -> int | None   # None = undecided
    def solve(self, cfg, puzzle, time_limit=None) -> solution | None
    def rate(self, cfg, puzzle) -> float                     # 0..10 (default: 5)
    # serialization (canonical JSON == what models see)
    def puzzle_to_json / puzzle_from_json / solution_to_json / solution_from_json
    # prompt fragments
    def prompt_rules(self, cfg) -> str
    def puzzle_json_schema(self, cfg) -> dict; def solution_json_schema(self, cfg) -> dict
    def prompt_make_tips(self, cfg) -> str; def prompt_solve_tips(self, cfg) -> str   # default ""
    # rendering
    def render_terminal(self, cfg, puzzle, solution=None, state=None) -> rich renderable | str
    def render_3d(self, cfg, puzzle, solution=None) -> plotly Figure      # optional
    # human play
    def human_help(self, cfg) -> str
    def new_state(self, cfg, puzzle); def parse_move(self, cfg, text) -> Move
    def apply_move(self, cfg, state, move) -> str; def state_to_solution(self, cfg, state)
```

Discovery (`engine/registry.py`):
1. every module `gamekhelega.games.<x>` that exports `GAME`;
2. every entry point in the group `gamekhelega.games`.
`get_game(name)` raises KeyError and lists what is available.

## 7. How to add a new game

1. Copy `examples/minimal_game.py` (an n×n Latin square, about 100 lines, every method implemented) to `src/gamekhelega/games/<name>/__init__.py`, or ship it in your own package and register it:
   ```toml
   [project.entry-points."gamekhelega.games"]
   mygame = "my_pkg.mygame:GAME"
   ```
2. Implement the methods in §6. The rules that matter:
   - `count_solutions(limit=2)` must be exact or return `None`. The referee never guesses.
   - The `*_to_json` output is exactly what models see. Keep it compact and human-readable.
   - `check_solution(..., reference=)` should fill `partial` ∈ [0,1], the fraction of free cells matching the reference.
   - `generate` must return a puzzle whose uniqueness you have proven.
3. `gamekhelega games` should list it. `gamekhelega generate -g <name> -p k=v` and a mock duel config with `"game": "<name>"` should work unchanged. `tests/test_engine.py::test_template_game_plugs_in` shows the minimum test.

## 8. Player adapters (`engine/adapters/`)

Every adapter implements `Player.act(Request) -> Reply(data, raw, error, usage)`. Specs come from the duel config:

| type | spec keys | notes |
|---|---|---|
| `mock` | id, skill (P(solve)), bad_make (P(corrupt a made puzzle)), difficulty, seed, make_time_budget | uses `game.generate`/`solve`; game-agnostic corruption (drops the solution) |
| `openai` | model, id, base_url, api_key_env, temperature, max_tokens, json_mode, retries, extra_body, extra_headers | stdlib urllib. Works with any `/chat/completions` server: OpenAI, OpenRouter, Together, vLLM, Ollama, LM Studio, xAI, etc. Uses `response_format: json_object`; this is dropped automatically if the server returns 400. Per-call timeout = time left. Retries on transport errors, then does one "repair" turn if the JSON does not parse. Sums token usage. |
| `command` | cmd (list or str), id, cwd, env | Request JSON goes on stdin; one JSON object is read from stdout. For local bots and solvers in any language. |
| `human` | id, make_mode (`assist` = engine generates at a chosen difficulty, `file` = load `{"puzzles":[...]}`) | SOLVE runs the interactive play loop |

## 9. Protocol and exact prompts (`engine/protocol.py`)

System prompt (every request):
> You are a contestant in a timed puzzle duel refereed by a program. Reply with exactly ONE JSON object and nothing else: no prose, no markdown, no code fences. It must match the JSON Schema you are given. Malformed output forfeits the item. The referee checks everything exactly.

The user prompt is built by `user_prompt(req)`:
```
GAME: <name>
CONFIG: <config json>

RULES:
<game.prompt_rules(cfg)>

TASK (MAKE): Create {count} puzzles that are as HARD as possible for a strong opponent to solve, but each
MUST have exactly one solution. For each puzzle also give its full solution. The referee rejects (and
penalises you for) any puzzle that is malformed, whose solution you give is wrong, or that has 0 or 2+ solutions.
Time limit: {T} s wall clock for this whole reply.
   -- or --
TASK (SOLVE): Solve puzzle {id}. It has exactly one solution.
Time limit: {T} s wall clock. If you cannot solve it, reply {"solution": null} (a wrong answer may earn
partial credit, a null earns nothing).

PUZZLE:
<puzzle json>

TIPS:
<game.prompt_make_tips / prompt_solve_tips>

OUTPUT JSON SCHEMA:
<schema>
```
Schemas:
- MAKE: `{"puzzles": [{"puzzle": <game puzzle schema>, "solution": <game solution schema>}, ...]}`
- SOLVE: `{"solution": <game solution schema> | null}`

The sudoku3d rules text is in `games/sudoku3d/plugin.py::prompt_rules`. Run `pytest tests/test_engine.py::test_prompts_and_json_extraction` to see it exercised. The MAKE tip gives a construction formula that has been verified valid for the layer rules: `((x-1) + b*((y-1) mod b) + (y-1) div b + (z-1)) mod n + 1`.

Parsing is lenient on purpose. `extract_json` strips code fences and leading prose, then takes the first `{...}`. Small models still score if they mostly follow the format.

## 10. Duel protocol (`engine/duel.py`)

```
DuelConfig: game, game_config, puzzles_per_side=3, make_time_s=600, solve_time_s=900,
            uniqueness_time_s=180, min_solve_request_s=5, seed, out_dir="runs", leaderboard
1. START   log config + player ids
2. MAKE    A and B in parallel: one MAKE request each (count = puzzles_per_side, limit = make_time_s)
3. REFEREE per submitted item -> check_submission:
             malformed (bad JSON/shape) | invalid_puzzle (givens break rules) |
             bad_solution (maker's solution wrong or doesn't match givens) |
             undecided (uniqueness not decided within uniqueness_time_s) | not_unique (0 or ≥2)
           accepted items get difficulty = game.rate(); missing items (fewer than asked) count as rejected
4. SOLVE   A and B in parallel; each side solves the OTHER side's accepted puzzles in order,
           sharing one budget solve_time_s; each request gets the time left; stop when < min_solve_request_s.
           Answers are checked with check_solution(reference = maker's solution) -> ok / partial.
5. SCORE   scoring.py (below), winner, tie-breaks
6. END     result.json, replay.jsonl, Elo update in leaderboard JSON
```

The referee measures time. Each call runs in a daemon thread and is abandoned after `limit + 1 s grace`; late replies are discarded and logged as `timeout`. The replay (`runs/<id>/replay.jsonl`) logs every request, raw reply (truncated), verdict, timing and token usage. `gamekhelega replay runs/<id> --boards` walks through it.

## 11. Scoring (`engine/scoring.py`)

Per side, with c = 1 if correct, p = partial ∈ [0,1], d = difficulty ∈ [0,10]:

```
solve_pts    = Σ_attempted [10·c + 3·(1−c)·p]  +  5 · n_correct · (1 − solve_time_used / solve_time_s)
make_pts     = Σ_own accepted [6·(1 − opponent_solved) + 4·d/10]  −  15 · n_own_rejected_or_missing
walkover_pts = 10 · n_opponent_rejected_or_missing     (you couldn't solve what they never validly gave)
total        = solve_pts + make_pts + walkover_pts
```
Winner = higher total. Ties are broken by more correct solves, then less solve time, then fewer rejected puzzles, then a draw.

Why: correctness dominates. Speed is a bonus only when you are correct. Making a puzzle that stumps the opponent is worth less than solving one (+6 vs +10), so "make impossible puzzles" is not a winning strategy. An invalid puzzle costs −15 for the maker and gives +10 to the opponent, so it is strictly worse than submitting an easy valid puzzle.

**Elo** (`engine/elo.py`): K=32, start 1500, stored per game: `{game: {player_id: {elo, w, l, d}}}`.

## 12. Visualization

- Terminal ✅ (`games/sudoku3d/render.py`, rich): layers shown side by side, with box separators. Givens are bold white, entries cyan, conflicts red, and empties `·`. The header shows filled, givens and conflicts. The human `view x|y|z [K]` command shows any slice.
- 3D 🟡 (plotly `Scatter3d`, `gamekhelega view pack.json --html out.html`): every cell is a marker with its symbol as text. Frames and a slider show one z-layer at a time or all layers. It works, but it needs polish (cloud agent): an "all layers" step, axis labels, colours for given vs filled, hover info, x/y-slice modes and an exploded-layer layout. The choice is plotly because it gives a self-contained HTML file with no server, works in notebooks, and is a 3D library the cloud agent already knows.
- Replay ✅ (terminal): `gamekhelega replay runs/<id> [--boards]`. HTML replay is optional (cloud agent).

## 13. Human mode

`gamekhelega play -g sudoku3d -p n=4 [--difficulty easy|medium|hard] [--puzzle pack.json --index 0] [--time-limit S]`, or `{"type": "human"}` in a duel config.

Game commands (sudoku3d; coordinates are 1-based):
- `set X Y Z V`, or just `X Y Z V`: put V at x=X y=Y z=Z.
- `clear X Y Z`: empty a cell (givens are locked).
- `view x|y|z [K]`: show slice K, or all slices.
- `undo`
- `check`: count conflicts and empties (no hints).

Engine commands: `help`, `submit` (or `s`), `giveup` (or `q`), `time`.

## 14. CLI

Every command takes `--game/-g`. For pack and run commands it overrides or asserts the game; for `generate`/`play` it is required, or comes from `GAMEKHELEGA_GAME`. Game config is passed with `--config '<json>'` and/or `-p key=value` (repeatable).

```
gamekhelega games
gamekhelega generate -g sudoku3d -p n=9 -p variant=layer --difficulty hard --count 5 --time-budget 60 --seed 1 --out pack.json
gamekhelega check pack.json            # verify rules + uniqueness + maker solutions
gamekhelega solve pack.json            # solve with the game's solver, report times
gamekhelega view pack.json [--index 0] [--solution] [--html cube.html]
gamekhelega play -g sudoku3d -p n=4 --difficulty easy
gamekhelega duel configs/duel_mock_4.json
gamekhelega replay runs/<id> [--boards]
gamekhelega leaderboard [-g sudoku3d]
```

## 15. zip3d ✅

Rules: one Hamiltonian path through an n³ grid (n = 3..6, default 4; 6-neighbour moves). It starts on
waypoint 1, ends on K, meets the waypoints 1..K in order and never crosses a wall. Duel puzzles must have
exactly one such path.

JSON (1-based coordinates):
- puzzle `{"waypoints": [[x,y,z,k],...], "walls": [[x,y,z,"+x"|"+y"|"+z"],...]}`; a wall `[x,y,z,"+x"]` blocks
  the face between (x,y,z) and (x+1,y,z) (`"-x"` etc. accepted and normalised)
- solution `{"path": [[x,y,z],...]}` with all n³ cells; `{"start": [x,y,z], "moves": "+x +y -z ..."}` also accepted

Config: `{"n": 4, "walls": false, "max_walls": -1}` (walls only affect the generator; submitted puzzles may
always contain walls; `max_walls = -1` means n). Unknown keys are rejected.

Files (`src/gamekhelega/games/zip3d/`):
- `model.py`: `ZipPuzzle(n, waypoints{cell: k}, walls{(a, b)})`, cell index `x + n·y + n²·z`, neighbour table.
- `solver_dfs.py`: iterative numba DFS. Prunes: global bipartite parity, waypoint order, unvisited cells
  stay connected, degree (every free non-end cell keeps ≥ 2 usable neighbours), the next waypoint stays
  reachable without stepping on later ones.
- `solver_cpsat.py`: CP-SAT `AddCircuit` over cells plus a dummy node (dummy → 1 and K → dummy fixed),
  rank variables (`rank[v] = rank[u] + 1` on used arcs) with strictly increasing waypoint ranks; counts up
  to `limit` by blocking found paths.
- `generator.py`: `count()` is the hybrid: DFS up to 20k nodes, then CP-SAT with the time limit (without
  ortools: DFS at ~1M nodes/s × limit). `generate()`: serpentine path → 20·N·⌊ln N⌋ backbite moves → N/8
  evenly spaced waypoints → repair (while an alternative path exists, pin the first cell where it diverges,
  or with `walls` sometimes wall that edge) → greedy waypoint removal down to a share of cells
  (easy 0.45, medium 0.30, hard 0 = locally minimal) → greedy wall removal. Undecided checks keep the clue.
- rating 0..10: `4·(1 − K/N) + 6·min(1, log10(1 + DFS nodes)/5)` with nodes capped at 1e6.
- `render.py`: rich layer panels (waypoints, step numbers, links, ↑/↓ for z moves, walls); plotly cube with
  the path as a line coloured by step, show/hide-solution buttons, slider `all` / `exploded` / layer 1..n.
- `plugin.py`: the `PuzzleGame`, prompts/schemas/tips, human mode (`+x -y`, `goto X Y Z` extends or cuts
  back, `undo`, `reset`, `view [K]`, `check`), partial credit = share of reference path edges present.

Measured (`research/zip3d_gen.log`, `research/zip3d_solvers.log`; 8 vCPU box):

| n | difficulty | budget | gen time median (max) | K median (range) | unique re-check |
|---|---|---|---|---|---|
| 3 | hard | 30 s | 0.01 s (0.12) | 9 (7-10) | 10/10 |
| 3 | hard + walls | 30 s | 0.01 s (0.11) | 8 (7-10), 1.5 walls | 10/10 |
| 3 | easy | 30 s | 0.01 s (0.15) | 12 (10-12) | 10/10 |
| 4 | hard | 60 s | 8.1 s (34.7) | 17.5 (16-22) | 8/8 |
| 4 | medium | 60 s | 5.6 s (11.3) | 19 (19-22) | 8/8 |
| 4 | hard + walls | 60 s | 10.3 s (17.2) | 16.5 (15-19), 2 walls | 6/6 |
| 5 | hard | 120 s | 126 s (127, budget hit) | 56.5 (50-77) | 4/4 |

Counting to 2 on ambiguous puzzles: DFS settles 3³ instantly and 4³ in ≤ 0.6 s (up to 1.2M nodes), but
aborts at 2M nodes on 5³/6³; CP-SAT takes 0.2 s (4³), 1.6-17 s (5³) and 43 s (6³, K = 60). Uniqueness
proofs on generated 4³ puzzles took 0.1-9 s.

Limits: 5³ generation runs out of budget during reduction, so its puzzles keep ~40-60 % of cells as
waypoints; 6³ uniqueness checks can exceed a 30 s referee limit (`undecided`). The rating saturates around
9 on 4³ because the DFS node count hits its cap. Duel configs therefore default to n = 4
(`configs/duel_mock_zip4.json`: make 90 s, solve 60 s, uniqueness 30 s).
