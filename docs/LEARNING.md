# Learning guide: the educational PR stack

This repo's open stack (PR #3 through this one) is a short course in building a
reliable referee. Each pull request is one lesson: a small, testable change, stacked
on the previous branch and **not** merged to `main`.

How to read it:

```bash
git fetch origin
git log --oneline --reverse origin/main..origin/edu/17-learning-guide
gh pr diff 3    # or any later number
```

There is no CI. After `pip install -e ".[dev]"`, run the tests locally with `pytest -q`
(first run is slower: numba warms up). Work through the sections in order; each "Try
it" should take a few minutes.

---

## #3 Seed the duel (`edu/01-seed`)

**Lesson.** A `seed` field that nothing reads is decoration. Reproducibility means
every RNG the run touches is derived from that seed — including the players.

**Read.** `DuelConfig.seed` and the `SeedSequence.spawn` call in
`src/gamekhelega/engine/duel.py`; `Player.set_seed` in
`src/gamekhelega/engine/adapters/base.py` and `MockPlayer.set_seed`.

**Try it.** Run `gamekhelega duel configs/duel_mock_4.json` twice with the same
`seed` and once with a different one. Compare `runs/*/result.json` puzzles and
scores. Then `pytest -q tests/test_engine.py -k seed`.

**Think.** If only the referee is seeded and the players keep their own RNGs, what
about the duel is still reproducible, and what isn't?

---

## #4 Atomic leaderboard writes (`edu/02-atomic-leaderboard`)

**Lesson.** Two processes writing the same JSON file can tear it or drop an update.
`open("w")` is not atomic. The usual pattern is: lock, write a temp file, `os.replace`.

**Read.** The lock and replace in `src/gamekhelega/engine/elo.py` (`O_CREAT|O_EXCL`
lockfile). The concurrent test in `tests/test_engine.py`.

**Try it.** Run `pytest -q tests/test_engine.py -k elo_concurrent`. Then imagine
removing the lock: what failure would you expect, and would it be deterministic?

**Think.** Why `os.replace` rather than `os.rename`, and why is the lock a separate
file instead of locking the JSON itself?

---

## #5 Rebuild the board from results (`edu/03-leaderboard-rebuild`)

**Lesson.** A derived cache should be rebuildable from the source of truth. If the
leaderboard is only a running total, a crash or a bad write is permanent.

**Read.** `elo.rebuild` in `src/gamekhelega/engine/elo.py` and the
`gamekhelega elo rebuild` command in `src/gamekhelega/cli.py`.

**Try it.** Run a couple of mock duels, delete or edit `runs/leaderboard.json`, then
rebuild from the `result.json` files. Confirm the Elo matches a fresh replay of the
same outcomes.

**Think.** What should happen if two `result.json` files disagree about the same
`duel_id`, or if a result is missing a winner?

---

## #6 Cancel work on timeout (`edu/04-cancel-timeouts`)

**Lesson.** Abandoning a thread at the deadline does not stop the work. An HTTP
request can keep running — and being billed — in the background. Cancellation has
to be a signal the adapter honours.

**Read.** `Request.cancel` in `src/gamekhelega/engine/protocol.py`, the deadline
path in `src/gamekhelega/engine/duel.py` (`_timed`), and how the OpenAI adapter
closes its socket.

**Try it.** `pytest -q tests/test_engine.py -k cancel`. Then read what happens if
`act()` ignores `cancel`: the referee returns a timeout, but the child is still
alive.

**Think.** Threads cannot be killed safely in Python. What can you cancel, and what
can you only abandon?

---

## #7 Scoring as properties (`edu/05-scoring-properties`)

**Lesson.** A comment like "an invalid puzzle is worse than a stump" is a
specification. Property tests (and a careful `None` vs `False`) stop a silent change
in who wins. `opponent_correct is None` is not a stump: the opponent never started.

**Read.** `tests/test_scoring_properties.py` and the `None` handling in
`src/gamekhelega/engine/scoring.py`. DESIGN §11's "Why" paragraph.

**Try it.** `pytest -q tests/test_scoring_properties.py`. Flip `W_INVALID` below
`W_STUMP + W_DIFF` and watch which test fails. Then set `opponent_correct` to
`None` in a one-off `score_side` call and confirm there is no stump bonus.

**Think.** Why would awarding stump points on a skip let a maker farm points by
running out the opponent's clock?

---

## #8 Best-of-N with swapped seats and a Wilson CI (`edu/06-match-stats`)

**Lesson.** A single duel is a noisy measurement. A match swaps seats so side A/B
is not a confound, and a Wilson interval around the win rate says how much to
trust a small `n`.

**Read.** `src/gamekhelega/engine/match.py` and `gamekhelega match` in
`src/gamekhelega/cli.py`.

**Try it.** `pytest -q tests/test_engine.py -k match`. Run a short mock match and
look at the CI in the summary: with `n=2`, how wide is it?

**Think.** Why swap seats rather than just play more duels with a fixed seating?
When is a Wilson interval more honest than "7 wins out of 10, so 70%"?

---

## #9 Parse JSON by scanning candidates (`edu/07-robust-json-extract`)

**Lesson.** Models wrap JSON in prose and fences. Taking the first `{` is fragile;
scanning `raw_decode` candidates, preferring a ` ```json ` block, and taking the
last object that has the expected keys is much more forgiving.

**Read.** `extract_json` in `src/gamekhelega/engine/protocol.py`.

**Try it.** `pytest -q tests/test_engine.py -k extract_json`. Feed `extract_json` a
string with a decoy `{...}` in the prose and a fenced block after it. Which object
wins, and why?

**Think.** When should you prefer the last matching object over the first? What
happens if two fenced blocks both look valid?

---

## #10 Retry with backoff (`edu/08-retry-backoff`)

**Lesson.** 429 and 5xx are transient; 401 and other 4xx are not. Retries need
backoff, jitter, and `Retry-After`. Usage must be summed across attempts, and a
local `json_mode` flag must not mutate the player's config.

**Read.** The retry loop in `src/gamekhelega/engine/adapters/openai_compat.py`.

**Try it.** `pytest -q tests/test_engine.py -k retry`. Trace what happens on a 401
versus a 429 with a `Retry-After` header.

**Think.** Why is mutating `player.json_mode` during a call a bug even if it makes
the next retry succeed?

---

## #11 Default `max_tokens` (`edu/09-default-max-tokens`)

**Lesson.** An unbounded completion is a blank cheque. A default cap (8192) is a
guardrail; raise it where the reply is legitimately large (the 10-puzzle 9³ example
uses 16384), and use `null` to send no cap.

**Read.** `DEFAULT_MAX_TOKENS` in `src/gamekhelega/engine/adapters/openai_compat.py`
and `configs/duel_llm_example.json`.

**Try it.** `pytest -q tests/test_engine.py -k max_tokens`. Check which request body
is sent when `max_tokens` is omitted, set to 16384, or set to `null`.

**Think.** Why isn't "no cap" the default? What is the failure mode if the cap is
silently too small for a MAKE of ten 9×9×9 cubes?

---

## #12 Cancel a command-player subprocess (`edu/10-command-cancel`)

**Lesson.** `subprocess.Popen` work lives in a process group. Honouring
`req.cancel` means killing that group (`killpg`), not just abandoning a `wait()`.
Also, `dataclasses.asdict` deep-copies a `Request` and blows up on an `Event`.

**Read.** `CommandPlayer` in `src/gamekhelega/engine/adapters/command.py` and
`Request.to_json` in `src/gamekhelega/engine/protocol.py`.

**Try it.** `pytest -q tests/test_engine.py -k command`. Start a dummy
`CommandPlayer` that sleeps, cancel it, and confirm the child is gone
(`/proc/<pid>` or `ps`).

**Think.** Why kill the process group rather than just the child pid? Why does
`Request.to_json` have to skip `cancel`?

---

## #13 Bradley-Terry from the event log (`edu/11-bradley-terry`)

**Lesson.** Elo is a sequential running total; Bradley-Terry fits a static strength
to the whole event log, and a bootstrap gives a confidence interval. The log is
the source of truth; the rating is a view of it.

**Read.** `src/gamekhelega/engine/bt.py` (Hunter MM + bootstrap) and
`gamekhelega elo bt` in `src/gamekhelega/cli.py`.

**Try it.** `pytest -q tests/test_engine.py -k bt`. Fit BT on a folder of mock
`result.json` files and compare the ranking to `gamekhelega leaderboard`.

**Think.** When would BT and Elo disagree on the ranking? What does a bootstrap
interval that crosses the opponent's rating tell you?

---

## #14 Elo keyed by game and size (`edu/12-leaderboard-by-game-size`)

**Lesson.** Strength is not a single number. n=4 and n=9 Sudoku are different
games for rating purposes. The unit of a leaderboard is `(game, size_key)`. Legacy
`{game: {player}}` files still have to be readable; the first write wraps them
under `""`.

**Read.** `size_key` and `record` in `src/gamekhelega/engine/elo.py`.

**Try it.** `pytest -q tests/test_engine.py -k elo_keys`. Build a tiny legacy
leaderboard JSON by hand, record one new result, and inspect how the file is
reshaped.

**Think.** Why is a missing size stored as `""` rather than guessed from the game's
default config?

---

## #15 Parallel uniqueness proofs (`edu/13-parallel-validation`)

**Lesson.** The GIL is a lock on the Python interpreter. Threads are fine for
waiting on I/O, but uniqueness proofs are CPU-bound SAT/search, so they need a
process pool. Use `spawn`, not `fork`: the parent may already have initialised
numba. Keep result order and the per-item time cap.

**Read.** `check_submissions` and `validation_pool` in
`src/gamekhelega/engine/validate.py`. `tests/conftest.py` sets
`GAMEKHELEGA_VALIDATION_WORKERS=1` so mock duels stay in-process.

**Try it.** `pytest -q tests/test_engine.py -k parallel_validation`. Then run the
same test with the cap in mind: does changing `validation_workers` change the
verdicts, or only the wall time?

**Think.** Why is forking after numba unsafe? Why would a default pool of 4 workers
make every mock test slower?

---

## #16 Bradley-Terry by game and size (`edu/14-bt-by-size`)

**Lesson.** The unit being rated has to be consistent across systems. Once Elo is
per `(game, size)`, BT has to be too — otherwise a player who is strong at n=4
and weak at n=9 looks average.

**Read.** `events_from_runs` / `fit_runs` in `src/gamekhelega/engine/bt.py`. The
`game_config` field now stored on match summaries.

**Try it.** `pytest -q tests/test_engine.py -k bt_ratings_are_per_game_and_size`.
The fixture has A win every n=4 duel and lose every n=9: confirm the two buckets
do not cancel.

**Think.** If a match file omitted `game_config`, what size should the events fall
into, and is that safer than guessing?

---

## #17 Named scoring weights (`edu/15-scoring-weights-config`)

**Lesson.** Magic numbers hide policy. `15` and `10` are the claim that an invalid
puzzle is worse than any valid one, and that a walkover does not erase that cost.
A dataclass makes the claim named, overridable, and testable.

**Read.** `ScoringWeights` in `src/gamekhelega/engine/scoring.py`, DESIGN §11, and
`test_default_ordering_invariants` in `tests/test_scoring_properties.py`.

**Try it.** `pytest -q tests/test_scoring_properties.py`. Override `invalid` in a
`DuelConfig.from_dict({..., "scoring": {"invalid": 1}})` and score one rejected
puzzle. Then lower the default `invalid` until the ordering test fails.

**Think.** If you raise `stump` above `solve`, what strategy does the game start to
reward, and should the tests block that or allow it as a deliberate policy change?

---

## #18 Slim core dependencies (`edu/16-slim-core-deps`)

**Lesson.** Optional dependencies and import-time cost. The referee should import
without pulling numba, python-sat, or ortools. Those belong in an extra, loaded
lazily, with a clear install error. Keep a dep at import only when the module
truly needs it (`@njit`, a SAT constructor).

**Read.** `pyproject.toml` (`[solvers]`), the lazy imports in the sudoku3d and
zip3d plugins, and `tests/test_optional_deps.py`.

**Try it.** `pytest -q tests/test_optional_deps.py`. In a fresh interpreter, block
`numba` (see that test) and `import gamekhelega.engine.duel` and
`all_games()`. Then import `gamekhelega.games.sudoku3d.solver_numba` and read the
error.

**Think.** Why is `@njit` a reason to keep numba at *module* import, but not a
reason to import that module when the plugin is only being registered?

---

## This PR: the stack as a course (`edu/17-learning-guide`)

**Lesson.** A stack of small PRs is already a curriculum if you write down what to
read, what to try, and what to ask. The document is part of the work, not an
afterthought.

**Read.** This file, and the README note that tests are local because there is no CI.

**Try it.** Walk #3 through #18 in order. For one PR, run only the tests named in
its "Try it" and then `gh pr diff N` to match the failure you would have seen
before the fix.

**Think.** If you added a #20, what would its lesson be, and how would you keep
the diff small enough to stay a single section?
