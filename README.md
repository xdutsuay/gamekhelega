# gamekhelega

*"game khele ga?" (will you play?)*: a game-agnostic arena for **AI-vs-AI (or human) puzzle duels**.

Each side **makes** puzzles for its opponent, then **solves** the opponent's puzzles under wall-clock limits.
A referee checks every puzzle for rule validity and a **unique** solution, grades the answers, scores the duel,
and updates per-game Elo ratings. Games are plugins: **3D Sudoku** (9×9×9: every line along x, y and z holds 1-9, and every layer is a
sudoku) and **3D Zip** (one path through every cell of an n×n×n cube, through numbered waypoints in order).

## Quickstart

```bash
pip install -e ".[dev]"             # Python >= 3.10
pytest -q                           # ~47 tests, ~20 s after numba warm-up
gamekhelega games
gamekhelega duel configs/duel_mock_4.json        # full duel with mock players, no API key
gamekhelega generate -g sudoku3d -p n=9 --difficulty medium --count 2 --out pack.json
gamekhelega check pack.json && gamekhelega view pack.json --solution --html cube.html
gamekhelega play -g sudoku3d -p n=4              # play in the terminal (human mode)
gamekhelega duel configs/duel_mock_zip4.json     # same, 3D Zip on a 4x4x4 cube
gamekhelega play -g zip3d -p n=3                 # 3D Zip in the terminal
gamekhelega leaderboard                          # Elo per game, from runs/leaderboard.json
```

A duel writes `runs/<id>/replay.jsonl` and `runs/<id>/result.json`; walk through one with
`gamekhelega replay runs/<id> --boards`.

`cube.html` is an interactive 3D view (plotly, loaded from a CDN): pick the slice axis (z-layers, y-slices,
x-slices) in the dropdown, then use the slider for the whole cube (`all`), an `exploded` view with the slices
spread apart, or a single slice. Givens are dark, solver-filled cells blue, empty cells faint grey; with
`--solution` a toggle shows or hides the filled cells. Hover shows `x, y, z = value (given|filled|empty)`.

## Running an LLM duel

Any OpenAI-compatible `/chat/completions` endpoint works (OpenAI, OpenRouter, Ollama, vLLM, LM Studio, ...).
`configs/duel_llm_example.json` pits a hosted model against a local Ollama model on the full 9×9×9 cube
(10 puzzles per side, 10 min to make, 15 min to solve):

```json
"player_a": {"type": "openai", "id": "openai:gpt-4o-mini", "model": "gpt-4o-mini",
             "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
"player_b": {"type": "openai", "id": "ollama:qwen2.5:7b", "model": "qwen2.5:7b",
             "base_url": "http://localhost:11434/v1", "api_key_env": null}
```

```bash
ollama pull qwen2.5:7b                     # local side (Ollama serves http://localhost:11434/v1)
export OPENAI_API_KEY=...                  # hosted side; the key is read from the env var named in api_key_env
gamekhelega duel configs/duel_llm_example.json
```

Swap a player without editing the file with `--a` / `--b` (a JSON player spec), e.g. OpenRouter:

```bash
export OPENROUTER_API_KEY=...
gamekhelega duel configs/duel_llm_example.json \
  --a '{"type": "openai", "id": "or:llama-3.1-8b", "model": "meta-llama/llama-3.1-8b-instruct",
        "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"}'
```

API keys are never written to logs or replays.

## Writing a bot

The `command` adapter runs your program once per request: the request JSON arrives on **stdin**, and your
reply JSON goes to **stdout** (extra text around the JSON object is tolerated). A non-zero exit, a timeout or
unparseable output counts as no answer.

Request fields (`engine/protocol.py::Request`):

| field | meaning |
|---|---|
| `phase` | `"make"` or `"solve"` |
| `game`, `config` | game name and its config, e.g. `"sudoku3d"`, `{"n": 9, "variant": "layer"}` |
| `rules` | the full rules text for this game and config |
| `time_limit_s` | wall-clock budget for this request (the referee measures it) |
| `count` | make: how many puzzles to produce |
| `puzzle_id`, `puzzle` | solve: the puzzle's id and its canonical JSON |
| `tips` | game-specific hints for this phase |
| `output_schema` | JSON schema your reply must match |

Expected replies:

```text
make:  {"puzzles": [{"puzzle": <puzzle JSON>, "solution": <solution JSON>}, ...]}   (at most `count`)
solve: {"solution": <solution JSON>}      or      {"solution": null}   to give up
```

For sudoku3d, puzzles and solutions are `{"layers": [...]}`: a list of z-layers, each a list of row strings
(`"0"` = empty, puzzles only), e.g. for n=4: `{"layers": [["1200", "0010", "0000", "4001"], ...]}`. Coordinates are 1-based everywhere models see them.

A minimal (and hopeless) bot, `bot.py`:

```python
import json, sys

req = json.load(sys.stdin)
print(json.dumps({"puzzles": []} if req["phase"] == "make" else {"solution": None}))
```

Pit it against a mock player:

```bash
gamekhelega duel configs/duel_mock_4.json --b '{"type": "command", "id": "my-bot", "cmd": ["python", "bot.py"]}'
```

## Adding a game

Games are plugins discovered from `gamekhelega.games.*` or the `gamekhelega.games` entry point; the engine
has no game-specific code. See [DESIGN.md §7](DESIGN.md#7-how-to-add-a-new-game) and the complete template
[examples/minimal_game.py](examples/minimal_game.py) (an n×n Latin square). `games/zip3d` is a second
full example that uses a different solver stack (DFS + CP-SAT) behind the same interface.

## 3D Zip (`zip3d`)

An n×n×n grid (n = 3..6, default 4) with numbered waypoints 1..K and optional walls. Draw **one** path that
visits every cell exactly once, moving only between face-adjacent cells (±x, ±y, ±z), starting on 1,
ending on K and meeting the waypoints in increasing order, never crossing a wall. Duel puzzles must have
exactly one such path.

```json
puzzle:   {"waypoints": [[1,1,1,1], [3,2,1,2], ...], "walls": [[2,1,1,"+x"], ...]}
solution: {"path": [[1,1,1], [2,1,1], ...]}          (all n³ cells; {"start":[x,y,z],"moves":"+x +y -z ..."} also accepted)
```

```bash
gamekhelega generate -g zip3d -p n=4 -p walls=true --difficulty hard --out zip.json
gamekhelega view zip.json --solution --html zip.html   # path as a 3D line coloured by step, layer slider
gamekhelega play -g zip3d -p n=3                         # moves: "+x -y +z", "goto X Y Z", undo, view 2, check
```

The referee proves uniqueness with a pruned depth-first search (numba) and hands hard cases to an OR-Tools
CP-SAT model (`AddCircuit` + order constraints), so `ortools` is a core dependency. Measured on the box
(`research/zip3d_gen.log`): 3³ puzzles generate in ~0.01 s, 4³ in ~5-35 s; 5³ is slow and 6³ uniqueness
proofs can time out (counted as `undecided`), so duels default to n = 4.

## Scoring

Per side, with c = 1 if correct, p = partial ∈ [0,1], d = difficulty ∈ [0,10]:

```
solve_pts    = Σ_attempted [10·c + 3·(1−c)·p]  +  5 · n_correct · (1 − solve_time_used / solve_time_s)
make_pts     = Σ_own accepted [6·(1 − opponent_solved) + 4·d/10]  −  15 · n_own_rejected_or_missing
walkover_pts = 10 · n_opponent_rejected_or_missing
total        = solve_pts + make_pts + walkover_pts
```

Winner = higher total; ties go to more correct solves, then less solve time, then fewer rejected puzzles.
Elo: K=32, start 1500, per game. Rationale and details: [DESIGN.md §11](DESIGN.md).

- Design, rules, protocol, prompts and benchmarks: [DESIGN.md](DESIGN.md)

License: MIT.
