"""Game-agnostic referee checks for maker submissions."""
from __future__ import annotations

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
