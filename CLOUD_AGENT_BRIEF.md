# Cloud agent brief: gamekhelega

You are finishing a small, already-working Python project. **Most of it is done and tested.**
Keep your token use low: read only the files a task names, and do not refactor working code.

## 0. Target repo and git rules (read first)

- Target: **public** GitHub repo **`xdutsuay/gamekhelega`**.
- If it does not exist, create it as **public** with no template, no auto README and no license. This pack already has them.
- **Initial commit:** push the pack contents, exactly as delivered, as the first commit on `main`. Message: `Initial import: engine, adapters, sudoku3d plugin, tests`.
  - If the repo already exists and `main` is non-empty, do NOT force-push. Open everything as a PR instead.
- Do all the remaining work below on branch **`feat/finish-v0.1`**, then open **one PR** into `main`. Title: `Finish v0.1: 3D viewer polish, CI, README`.
- **Never merge** the PR, never push to `main` after the initial commit, and never force-push.
- Public repo: commit no secrets, API keys, `.env` files, `runs/` output or personal info. `.gitignore` already covers `.venv/`, `runs/` and caches.
- Use cheap and fast models. Do not run LLM duels against paid APIs. Tests and the mock duel need no keys.

## 1. Current state (all verified on a Linux box, Python 3.13)

```
pip install -e ".[dev]"
pytest -q                     # 28 passed (a few seconds once numba has compiled)
gamekhelega duel configs/duel_mock_4.json   # full mock duel -> runs/<id>/{replay.jsonl,result.json}, Elo in runs/leaderboard.json
```

Done and tested (do not rewrite):

| area | files | tests |
|---|---|---|
| plugin interface + registry (built-in + entry points `gamekhelega.games`) | `src/gamekhelega/engine/game.py`, `registry.py` | `test_registry_discovers_games`, `test_template_game_plugs_in` |
| engine has zero game-specific code | `src/gamekhelega/engine/**` | `test_engine_has_no_game_specific_code` |
| protocol: exact prompts, schemas, lenient JSON | `engine/protocol.py` | `test_prompts_and_json_extraction` |
| referee validation (malformed / invalid / bad_solution / undecided / not_unique) | `engine/validate.py` | `test_check_submission_reasons` |
| duel runner, hard timeouts, replay JSONL, Elo | `engine/duel.py`, `scoring.py`, `elo.py` | `test_mock_duel_end_to_end`, `test_scoring_and_winner`, `test_elo` |
| adapters: mock, OpenAI-compatible HTTP, command, human | `engine/adapters/*` | `test_openai_adapter_against_fake_server`, `test_command_adapter`, `test_human_adapter_scripted` |
| sudoku3d plugin: rules (layer + symmetric, 4³/9³), constructions, numba + SAT hybrid uniqueness, generator, rating, JSON, human moves, terminal render | `src/gamekhelega/games/sudoku3d/*` | `tests/test_sudoku3d.py` (14 tests) |
| CLI: games, generate, solve, check, play, duel, view, replay, leaderboard (all take `--game`) | `src/gamekhelega/cli.py` | `tests/test_cli.py` |
| template game (Latin square) | `examples/minimal_game.py` | `test_template_game_plugs_in` |
| zip3d plugin: path model, numba DFS + CP-SAT uniqueness, generator (backbite + waypoint repair/reduction, optional walls), rating, JSON, human moves, terminal + plotly render | `src/gamekhelega/games/zip3d/*` | `tests/test_zip3d.py` |

Rough, with work left for you: the plotly 3D view, `games/sudoku3d/render.py::render_3d`.

## 2. Remaining tasks (do them in this order; each has acceptance criteria)

### T1. Bootstrap (≈5 min)
- Create, push and branch as described in §0.
- **Accept:**
  - `main` holds the pack as delivered.
  - Branch `feat/finish-v0.1` exists.
  - `pip install -e ".[dev]" && pytest -q` passes on a clean checkout.

### T2. CI (`.github/workflows/ci.yml` already exists as a stub)
- Make it run on Python 3.10, 3.11 and 3.12 (ubuntu-latest), with pip cache. Steps: `pip install -e ".[dev]"`, `pytest -q`, then a smoke run of `gamekhelega duel configs/duel_mock_4.json`.
- Set the `NUMBA_CACHE_DIR` env var to a temp dir.
- **Accept:**
  - The workflow is green on the PR.
  - The total job time is under 10 min.
  - If numba or python-sat wheels are missing for a Python version, drop that version and say so in the PR description. Do not add hacks.

### T3. 3D viewer polish (`src/gamekhelega/games/sudoku3d/render.py::render_3d` only)
Current state: Scatter3d markers with text, frames per z-layer, and a slider. The "all layers" step is overwritten, so it is broken.
- Slider steps: `all`, then `z=1..n`. Add a dropdown (updatemenus) for slice axis: z-layers, y-slices, x-slices. Within that axis, show only one slice, all slices, or an "exploded" view (slices offset by a gap along the axis).
- Colours: givens in dark/bold, solver-filled cells in a different colour, and empties as faint small markers. When the solution is passed, add a toggle for showing or hiding the filled cells.
- Hover shows `x, y, z = value (given|filled|empty)`, and axes are labelled x, y, z with 1-based ticks.
- Keep it self-contained, in one HTML file via `fig.write_html(..., include_plotlyjs="cdn")`. This is already done in `engine/viewer.py::export_html`; do not change the engine.
- **Accept:**
  - `gamekhelega view <pack> --solution --html cube.html` works for n=4 and n=9.
  - `tests/test_sudoku3d.py::test_renderers` still passes. Update its `len(fig.frames)` assertion if the frame count changes, and add a test that the n=9 figure has ≥ 10 slider steps and an updatemenus axis selector.
  - The HTML file is under 5 MB for n=9.

### T4. README polish
- Text only (no GIFs or images). Add these sections:
  - Quickstart (already there).
  - "Running an LLM duel", with an example using a local Ollama base URL plus one hosted endpoint, read from `configs/duel_llm_example.json`.
  - "Writing a bot", covering the command adapter's stdin/stdout JSON. Show the Request fields from `engine/protocol.py::Request` and the expected reply schemas.
  - "Adding a game", linking DESIGN.md §7.
  - Scoring in brief (copy the formula from DESIGN.md §11).
  - A CI badge for `xdutsuay/gamekhelega`.
- **Accept:**
  - Every command in the README runs as written on a clean install. Verify by running them.

### T5. Open the PR
- Open the PR with a description listing T2-T4, the test output and anything you skipped and why.
- **Do not merge.**

## 3. Out of scope (do NOT do)
- Change scoring constants or prompts, swap solvers, add web servers or databases, or add dependencies beyond plotly (already an extra) and ortools (core since zip3d).
- Run benchmarks. Numbers are in DESIGN.md §5 and `research/*.log`.
- Rename or reorganize the package.

## 4. Handy facts
- Coordinates in JSON and prompts are 1-based. Internal arrays are `g[z, y, x]`, 0-based, with `0` = empty.
- Sudoku3d config is `{"n": 4|9, "variant": "layer"|"symmetric"}`, and the canonical variant is `layer`.
- The numba first-call JIT takes about 5-10 s, and is cached in `__pycache__` afterwards.
- If anything in §1 fails on a clean checkout, fix the smallest thing possible and note it in the PR.
- The referee's uniqueness proof can take 30-90 s on hard 9³ layer puzzles (pysat, cold). The default `uniqueness_time_s` is 180. Do not lower it in CI configs other than `duel_mock_4.json`.
