"""Summaries of results/leaderboards (game-agnostic, rich tables)."""
from __future__ import annotations

import json
from pathlib import Path

from rich.table import Table


def result_table(result: dict) -> Table:
    t = Table(title=f"{result['game']} duel {result['duel_id']}  winner: {result.get('winner_id') or 'draw'}")
    # transposed (metric rows, one column per side) so it fits an 80-column terminal
    sides = list(result["scores"])
    t.add_column("metric")
    for k in sides:
        t.add_column(f"{k}: {result['players'][k]}", justify="right")
    cols = ["total", "solve_pts", "make_pts", "walkover_pts", "n_correct", "n_attempted",
            "solve_time_s", "n_rejected", "n_stumped"]
    for c in cols:
        t.add_row(c, *[str(result["scores"][k][c]) for k in sides])
    return t


def leaderboard_table(path: str, game: str | None = None) -> Table:
    data = json.loads(Path(path).read_text()) if Path(path).exists() else {}
    t = Table(title=f"Leaderboard ({path})")
    for c in ("game", "player", "elo", "w", "l", "d"):
        t.add_column(c)
    for g, board in sorted(data.items()):
        if game and g != game:
            continue
        for pid, r in sorted(board.items(), key=lambda kv: -kv[1]["elo"]):
            t.add_row(g, pid, f"{r['elo']:.1f}", str(r["w"]), str(r["l"]), str(r["d"]))
    return t
