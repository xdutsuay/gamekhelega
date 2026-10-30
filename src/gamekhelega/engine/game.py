"""The PuzzleGame plugin interface. The engine only ever talks to games via this.

A game plugin is a module exposing ``GAME: PuzzleGame`` (an instance), registered
either as a built-in subpackage of ``gamekhelega.games`` or via the entry-point
group ``gamekhelega.games`` (``name = "pkg.module:GAME"``).

Opaque types: a game's ``Puzzle``/``Solution``/``Config``/``State`` objects can be
anything; the engine only stores them via ``*_to_json`` / ``*_from_json``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np

C = TypeVar("C")  # game config
P = TypeVar("P")  # puzzle
S = TypeVar("S")  # solution


@dataclass
class CheckResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    partial: float = 0.0  # fraction in [0, 1] of the solution that is right (partial credit)


@dataclass
class Move:
    """Parsed human command. kind: game-defined ("set", "clear", "view", ...) or
    engine-reserved ("help", "submit", "quit", "undo")."""
    kind: str
    args: dict[str, Any] = field(default_factory=dict)


class PuzzleGame(ABC, Generic[C, P, S]):
    name: str = ""          # registry key, used by --game
    title: str = ""         # human readable
    description: str = ""

    # ---- config -------------------------------------------------------
    @abstractmethod
    def config_schema(self) -> dict:
        """JSON Schema for this game's config block (validated by the engine)."""

    @abstractmethod
    def parse_config(self, d: dict | None) -> C:
        """dict (may be partial/None) -> config object with defaults applied."""

    @abstractmethod
    def config_to_json(self, cfg: C) -> dict: ...

    # ---- core ----------------------------------------------------------
    @abstractmethod
    def generate(self, cfg: C, difficulty: str, rng: np.random.Generator,
                 time_budget: float | None = None) -> tuple[P, S]:
        """Return (puzzle, its unique solution). difficulty in {"easy","medium","hard"}."""

    @abstractmethod
    def validate_puzzle(self, cfg: C, puzzle: P) -> list[str]:
        """Well-formedness only (shape, value ranges, no direct clue conflicts).
        Empty list = OK. Uniqueness is checked separately by count_solutions."""

    @abstractmethod
    def check_solution(self, cfg: C, puzzle: P, solution: S,
                       reference: S | None = None) -> CheckResult:
        """Does `solution` satisfy all rules and agree with the puzzle's givens?
        If `reference` (the known unique solution) is given, set `partial` to the
        fraction of the solver's work that matches it (game-defined)."""

    @abstractmethod
    def count_solutions(self, cfg: C, puzzle: P, limit: int = 2,
                        time_limit: float | None = None) -> int | None:
        """Number of solutions capped at `limit`; None if undecided within time_limit."""

    @abstractmethod
    def solve(self, cfg: C, puzzle: P, time_limit: float | None = None) -> S | None:
        """Reference solver (used by the mock adapter and hints)."""

    def rate(self, cfg: C, puzzle: P) -> float:
        """Difficulty 0..10 (engine uses it in make-scoring). Default: 5."""
        return 5.0

    # ---- serialization (canonical JSON, also what models see/return) --
    @abstractmethod
    def puzzle_to_json(self, cfg: C, puzzle: P) -> Any: ...

    @abstractmethod
    def puzzle_from_json(self, cfg: C, data: Any) -> P:
        """Raise ValueError with a helpful message on malformed input."""

    @abstractmethod
    def solution_to_json(self, cfg: C, solution: S) -> Any: ...

    @abstractmethod
    def solution_from_json(self, cfg: C, data: Any) -> S: ...

    # ---- prompts (fragments; the engine wraps them) ---------------------
    @abstractmethod
    def prompt_rules(self, cfg: C) -> str:
        """Plain-text rules + coordinate convention + data format with a tiny example."""

    @abstractmethod
    def puzzle_json_schema(self, cfg: C) -> dict: ...

    @abstractmethod
    def solution_json_schema(self, cfg: C) -> dict: ...

    def prompt_make_tips(self, cfg: C) -> str:
        """Optional game-specific advice for the MAKE phase."""
        return ""

    def prompt_solve_tips(self, cfg: C) -> str:
        return ""

    # ---- rendering -----------------------------------------------------
    def render_terminal(self, cfg: C, puzzle: P, solution: S | None = None,
                        state: Any = None) -> Any:
        """Return a `rich` renderable (or str). Highlight givens vs filled vs errors."""
        raise NotImplementedError

    def render_3d(self, cfg: C, puzzle: P, solution: S | None = None) -> Any:
        """Return a plotly Figure (engine writes HTML)."""
        raise NotImplementedError

    # ---- human mode ----------------------------------------------------
    def human_help(self, cfg: C) -> str:
        raise NotImplementedError

    def new_state(self, cfg: C, puzzle: P) -> Any:
        """Mutable play state (current partial solution, view axis/layer, history)."""
        raise NotImplementedError

    def parse_move(self, cfg: C, text: str) -> Move:
        """Parse a typed command into a Move; raise ValueError with usage on error."""
        raise NotImplementedError

    def apply_move(self, cfg: C, state: Any, move: Move) -> str:
        """Mutate state; return a short status line."""
        raise NotImplementedError

    def state_to_solution(self, cfg: C, state: Any) -> S:
        raise NotImplementedError
