# gamekhelega

*"game khele ga?" (will you play?)*: a game-agnostic arena for **AI-vs-AI (or human) puzzle duels**.

Each side **makes** puzzles for its opponent, then **solves** the opponent's puzzles under wall-clock limits.
A referee checks every puzzle for rule validity and a **unique** solution, grades the answers, scores the duel,
and updates per-game Elo ratings. Games are plugins. The first one is **3D Sudoku** (9×9×9: every line along
x, y and z holds 1-9, and every layer is a sudoku).

```bash
pip install -e ".[dev]"             # Python >= 3.10
pytest -q                           # ~30 tests, a few seconds after numba warm-up
gamekhelega games
gamekhelega duel configs/duel_mock_4.json        # full duel with mock players, no API key
gamekhelega generate -g sudoku3d -p n=9 --difficulty medium --count 2 --out pack.json
gamekhelega check pack.json && gamekhelega view pack.json --html cube.html
gamekhelega play -g sudoku3d -p n=4              # play in the terminal
```

LLM duels work with any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, vLLM, ...). See
`configs/duel_llm_example.json`. API keys are read from the environment variable named in `api_key_env`
and are never written to logs.

- Design, rules, protocol, prompts, scoring and benchmarks: [DESIGN.md](DESIGN.md)
- Add your own game: DESIGN.md §7 and [examples/minimal_game.py](examples/minimal_game.py)

License: MIT.
