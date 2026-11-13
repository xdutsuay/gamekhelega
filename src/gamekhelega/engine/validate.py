"""Game-agnostic referee checks for maker submissions.

Uniqueness proofs are CPU-bound (SAT / search). Threads share the GIL, so
a pool of threads would not run those proofs in parallel. A process pool
does. We use the "spawn" start method because forking after numba or other
native libraries have been initialised in the parent is unsafe.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .game import PuzzleGame


@dataclass
class SubmissionVerdict:
    accepted: bool
    reason: str = ""  # "", "malformed", "invalid_puzzle", "bad_solution", "not_unique", "undecided"
    errors: list[str] = field(default_factory=list)
    difficulty: float = 0.0
    puzzle: Any = None
    solution: Any = None


def default_workers() -> int:
    env = os.environ.get("GAMEKHELEGA_VALIDATION_WORKERS")
    if env is not None and env.strip() != "":
        return max(1, int(env))
    return max(1, min(4, os.cpu_count() or 1))


def check_submission(game: PuzzleGame, cfg: Any, item: dict,
                     uniqueness_time_limit: float = 20.0) -> SubmissionVerdict:
    """A maker item is {"puzzle": <json>, "solution": <json>}. Accepted iff:
    parses, puzzle well-formed, maker's solution is valid for it, and the puzzle
    has exactly one solution (decided within the time limit; undecided = reject).
    """
    try:
        puzzle = game.puzzle_from_json(cfg, item["puzzle"])
        solution = game.solution_from_json(cfg, item["solution"])
    except (KeyError, TypeError, ValueError) as e:
        return SubmissionVerdict(False, "malformed", [str(e)])
    errs = game.validate_puzzle(cfg, puzzle)
    if errs:
        return SubmissionVerdict(False, "invalid_puzzle", errs, puzzle=puzzle)
    chk = game.check_solution(cfg, puzzle, solution)
    if not chk.ok:
        return SubmissionVerdict(False, "bad_solution", chk.errors, puzzle=puzzle)
    k = game.count_solutions(cfg, puzzle, limit=2, time_limit=uniqueness_time_limit)
    if k is None:
        return SubmissionVerdict(False, "undecided", ["uniqueness not decided in time"], puzzle=puzzle)
    if k != 1:
        return SubmissionVerdict(False, "not_unique", [f"{k}+ solutions"], puzzle=puzzle)
    return SubmissionVerdict(True, "", [], game.rate(cfg, puzzle), puzzle, solution)


def validation_pool(workers: int | None = None):
    """A spawn-context process pool, or None when work should stay in-process."""
    n = default_workers() if workers is None else max(1, int(workers))
    if n <= 1:
        return _NullPool()
    import multiprocessing as mp
    return ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))


class _NullPool:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False


def check_submissions(game: PuzzleGame, cfg: Any, items: list,
                      uniqueness_time_limit: float = 20.0,
                      workers: int | None = None,
                      pool: ProcessPoolExecutor | None = None) -> list[SubmissionVerdict]:
    """Check many maker items, keeping input order. Each item still has its
    own uniqueness time cap. `workers` defaults to min(4, cpu_count()); 1
    (or a single item) stays in-process so tests and tiny packs stay cheap.
    Pass an existing `pool` to reuse processes across two sides of a duel.
    """
    packed = [it if isinstance(it, dict) else {} for it in items]
    if not packed:
        return []
    n = default_workers() if workers is None else max(1, int(workers))
    if pool is None and (n == 1 or len(packed) == 1):
        return [check_submission(game, cfg, it, uniqueness_time_limit) for it in packed]
    name, config = game.name, game.config_to_json(cfg)

    def submit(executor):
        futs = [executor.submit(_check_one, name, config, it, uniqueness_time_limit)
                for it in packed]
        return [f.result() for f in futs]

    if pool is not None:
        return submit(pool)
    with validation_pool(n) as executor:
        return submit(executor)


def _check_one(game_name: str, game_config: dict, item: dict,
               uniqueness_time_limit: float) -> SubmissionVerdict:
    """Top-level worker target (must be picklable). Reloads the game in the child."""
    from .registry import get_game
    game = get_game(game_name)
    cfg = game.parse_config(game_config)
    return check_submission(game, cfg, item, uniqueness_time_limit)
