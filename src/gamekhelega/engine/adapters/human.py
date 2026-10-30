"""Human player in the terminal. MAKE: generate with engine help or load a file.
SOLVE: interactive play loop driven by the game's parse_move/apply_move/render."""
from __future__ import annotations

import json
import time

import numpy as np

from ..protocol import Request
from .base import Player, Reply

ENGINE_HELP = ("engine commands: help | submit | giveup | time\n")


class HumanPlayer(Player):
    def __init__(self, game, id: str = "human", input_fn=input, console=None,
                 make_mode: str = "assist"):
        """make_mode: "assist" = engine generates at a chosen difficulty;
        "file" = ask for a JSON file path with {"puzzles":[...]}."""
        self.game, self.id, self.input, self.make_mode = game, id, input_fn, make_mode
        if console is None:
            from rich.console import Console
            console = Console()
        self.console = console

    def act(self, req: Request) -> Reply:
        g, cfg = self.game, self.game.parse_config(req.config)
        if req.phase == "make":
            if self.make_mode == "file":
                path = self.input(f"path to JSON with {req.count} puzzles: ").strip()
                with open(path) as f:
                    return Reply(data=json.load(f))
            diff = self.input("difficulty for your puzzles [easy/medium/hard]: ").strip() or "medium"
            rng = np.random.default_rng()
            items = []
            for _ in range(req.count):
                p, s = g.generate(cfg, diff, rng, time_budget=req.time_limit_s / max(1, req.count) * 0.8)
                items.append({"puzzle": g.puzzle_to_json(cfg, p), "solution": g.solution_to_json(cfg, s)})
            return Reply(data={"puzzles": items})
        puzzle = g.puzzle_from_json(cfg, req.puzzle)
        sol = play_loop(g, cfg, puzzle, self.input, self.console, req.time_limit_s)
        return Reply(data={"solution": None if sol is None else g.solution_to_json(cfg, sol)})


def play_loop(game, cfg, puzzle, input_fn, console, time_limit_s: float | None = None):
    """Shared by `play` and the human duel adapter. Returns a solution or None."""
    state = game.new_state(cfg, puzzle)
    t0 = time.monotonic()
    console.print(game.human_help(cfg) + "\n" + ENGINE_HELP)
    while True:
        console.print(game.render_terminal(cfg, puzzle, state=state))
        left = "" if time_limit_s is None else f" [{time_limit_s - (time.monotonic() - t0):.0f}s left]"
        if time_limit_s is not None and time.monotonic() - t0 > time_limit_s:
            console.print("time up")
            return game.state_to_solution(cfg, state)
        try:
            text = input_fn(f">{left} ").strip()
        except EOFError:
            return None
        if text in ("submit", "s"):
            return game.state_to_solution(cfg, state)
        if text in ("giveup", "quit", "q"):
            return None
        if text == "help":
            console.print(game.human_help(cfg) + "\n" + ENGINE_HELP)
            continue
        if text == "time":
            console.print(f"{time.monotonic() - t0:.0f}s elapsed")
            continue
        try:
            msg = game.apply_move(cfg, state, game.parse_move(cfg, text))
        except ValueError as e:
            msg = f"[red]{e}[/red]"
        console.print(msg)
