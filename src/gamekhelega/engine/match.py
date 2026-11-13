"""Best-of-N matches: swap seats, vary seeds, report a win rate with a CI.

A single duel is a noisy sample. One lucky stump, or who sat on the left,
can swing Elo by a full K=32 step. A match repeats the pairing so the
number you quote comes with a confidence interval.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .adapters.base import Player
from .duel import DuelConfig, run_duel
from .game import PuzzleGame

Z95 = 1.96  # ~95% normal quantile for the Wilson interval


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n Bernoulli trials.

    Unlike a naive p ± 1.96·sqrt(p(1-p)/n), this stays inside [0, 1] and
    is honest about small n: 1 win in 1 game is about [0.21, 1.00], not
    [1.00, 1.00].
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def run_match(game: PuzzleGame, a: Player, b: Player, dc: DuelConfig,
              games: int = 5) -> dict:
    """Run `games` duels, swapping seats each time and incrementing the seed."""
    if games < 1:
        raise ValueError("games must be >= 1")
    match_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    records = []
    for i in range(games):
        swap = i % 2 == 1
        left, right = (b, a) if swap else (a, b)
        res = run_duel(game, left, right, replace(dc, seed=dc.seed + i))
        if res["winner"] == "draw":
            score_a = 0.5
        elif swap:
            score_a = 0.0 if res["winner"] == "a" else 1.0
        else:
            score_a = 1.0 if res["winner"] == "a" else 0.0
        records.append({
            "duel_id": res["duel_id"], "seed": dc.seed + i, "swapped": swap,
            "seats": {"a": left.id, "b": right.id},
            "winner": res["winner"], "winner_id": res.get("winner_id"),
            "score_a": score_a, "out_dir": res.get("out_dir"),
        })
    a_wins = sum(1 for r in records if r["score_a"] == 1.0)
    b_wins = sum(1 for r in records if r["score_a"] == 0.0)
    draws = sum(1 for r in records if r["score_a"] == 0.5)
    rate = (a_wins + 0.5 * draws) / games
    # Wilson is binomial; apply it to A's decisive-win rate (draws are not wins).
    # The printed rate still includes draws as half a win.
    ci = wilson_interval(a_wins, games)
    summary = {
        "match_id": match_id, "games": games, "game": game.name,
        "players": {"a": a.id, "b": b.id},
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_rate": round(rate, 3),
        "ci95": [round(ci[0], 3), round(ci[1], 3)],
        "duels": records,
    }
    out = Path(dc.out_dir) / f"match-{match_id}.json"
    Path(dc.out_dir).mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    summary["out"] = str(out)
    return summary
