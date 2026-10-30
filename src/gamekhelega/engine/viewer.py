"""Game-agnostic viewer shell: terminal render, 3D HTML export, replay walkthrough."""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from .game import PuzzleGame
from .registry import get_game


def show(game: PuzzleGame, cfg, puzzle, solution=None, console: Console | None = None) -> None:
    (console or Console()).print(game.render_terminal(cfg, puzzle, solution))


def export_html(game: PuzzleGame, cfg, puzzle, solution, path: str) -> str:
    fig = game.render_3d(cfg, puzzle, solution)
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def replay(run_dir: str, console: Console | None = None, show_boards: bool = False) -> None:
    """Print a duel's events in order. With show_boards, render each solve attempt."""
    con = console or Console()
    events = [json.loads(l) for l in (Path(run_dir) / "replay.jsonl").read_text().splitlines() if l]
    start = events[0]
    game = get_game(start["game"])
    cfg = game.parse_config(start["config"])
    result = json.loads((Path(run_dir) / "result.json").read_text())
    puzzles = {p["id"]: p for side in result["puzzles"].values() for p in side}
    for e in events:
        t = f"[dim]{e['t']:8.2f}s[/dim]"
        if e["event"] == "start":
            con.print(f"{t} [bold]{e['game']}[/bold] {e['players']['a']} vs {e['players']['b']}")
        elif e["event"] == "make_reply":
            con.print(f"{t} {e['side']} made {e['n_items']} puzzles in {e['time_s']}s {e['error']}")
        elif e["event"] == "verdict":
            mark = "[green]accepted[/green]" if e["accepted"] else f"[red]rejected: {e['reason']}[/red]"
            con.print(f"{t} puzzle {e['id']} {mark} difficulty={e['difficulty']}")
        elif e["event"] == "solve":
            mark = "[green]correct[/green]" if e["correct"] else f"[red]wrong[/red] partial={e['partial']:.2f}"
            con.print(f"{t} {e['side']} on {e['id']}: {mark} in {e['time_s']}s {e['error']}")
            if show_boards and e.get("solution") is not None:
                p = puzzles[e["id"]]
                con.print(game.render_terminal(cfg, game.puzzle_from_json(cfg, p["puzzle"]),
                                               game.solution_from_json(cfg, e["solution"])))
        elif e["event"] == "end":
            con.print(f"{t} winner: [bold]{e['winner']}[/bold] {e['scores']}")
