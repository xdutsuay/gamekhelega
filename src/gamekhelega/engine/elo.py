"""Per-game Elo leaderboard stored as JSON: {game: {player_id: {"elo":..,"w":..,"l":..,"d":..}}}."""
from __future__ import annotations

import json
import os
from pathlib import Path

K, BASE = 32.0, 1500.0


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update(ra: float, rb: float, score_a: float, k: float = K) -> tuple[float, float]:
    ea = expected(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


def record(path: str | os.PathLike, game: str, a: str, b: str, result: str) -> dict:
    """result in {"a","b","draw"}. Returns the updated board for `game`."""
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    board = data.setdefault(game, {})
    for pid in (a, b):
        board.setdefault(pid, {"elo": BASE, "w": 0, "l": 0, "d": 0})
    sa = {"a": 1.0, "b": 0.0, "draw": 0.5}[result]
    board[a]["elo"], board[b]["elo"] = (round(x, 1) for x in update(board[a]["elo"], board[b]["elo"], sa))
    if result == "draw":
        board[a]["d"] += 1; board[b]["d"] += 1
    else:
        w, l = (a, b) if result == "a" else (b, a)
        board[w]["w"] += 1; board[l]["l"] += 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return board
