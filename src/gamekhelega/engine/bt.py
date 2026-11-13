"""Bradley-Terry ratings from a batch of pairwise results.

Online Elo is path-dependent: the same set of duels, in a different
order, produces different ratings. Bradley-Terry fits one strength per
player to all the results at once (MM iterations, Hunter 2004) and a
bootstrap gives a confidence interval. The rating is reported on the
Elo scale (1500 + 400·log10(π_i / geomean(π))) so the two can be compared.

Events are bucketed by game and size (the same unit as the Elo leaderboard).
A 4-cube result and a 9-cube result do not share a rating.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .elo import size_key

BASE = 1500.0


def fit(events: list[tuple[str, str, float]], n_iter: int = 1000, tol: float = 1e-9):
    """events: (player_a, player_b, score_a) with score_a in {0, 0.5, 1}.

    Returns (players, pi, elo_ratings) or empty arrays if there are no events.
    """
    players = sorted({p for a, b, _ in events for p in (a, b)})
    if not players:
        return [], np.array([]), np.array([])
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    wins = np.zeros(n)
    games = np.zeros((n, n))
    for a, b, sa in events:
        i, j = idx[a], idx[b]
        if i == j:
            continue
        wins[i] += sa
        wins[j] += 1.0 - sa
        games[i, j] += 1.0
        games[j, i] += 1.0
    pi = np.ones(n)
    for _ in range(n_iter):
        denom = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j or games[i, j] == 0:
                    continue
                denom[i] += games[i, j] / (pi[i] + pi[j])
        new = np.divide(wins, denom, out=np.zeros(n), where=denom > 0)
        mean = new.mean()
        if mean > 0:
            new = new / mean
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    logp = np.log10(np.maximum(pi, 1e-15))
    logp -= logp.mean()
    return players, pi, BASE + 400.0 * logp


def bootstrap(events: list[tuple[str, str, float]], n_boot: int = 400,
              seed: int = 0, alpha: float = 0.05) -> dict[str, dict]:
    """Point ratings plus a percentile CI from resampling the event list."""
    players, _, ratings = fit(events)
    out = {p: {"rating": round(float(r), 1), "lo": None, "hi": None,
               "w": 0, "l": 0, "d": 0} for p, r in zip(players, ratings)}
    for a, b, sa in events:
        if sa == 0.5:
            out[a]["d"] += 1
            out[b]["d"] += 1
        elif sa == 1.0:
            out[a]["w"] += 1
            out[b]["l"] += 1
        else:
            out[a]["l"] += 1
            out[b]["w"] += 1
    if not events or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    ev = list(events)
    samples: dict[str, list[float]] = {p: [] for p in players}
    for _ in range(n_boot):
        pick = [ev[i] for i in rng.integers(0, len(ev), len(ev))]
        ps, _, rs = fit(pick)
        got = dict(zip(ps, rs))
        for p in players:
            if p in got:
                samples[p].append(float(got[p]))
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    for p, xs in samples.items():
        if xs:
            out[p]["lo"] = round(float(np.percentile(xs, lo_q)), 1)
            out[p]["hi"] = round(float(np.percentile(xs, hi_q)), 1)
    return out


def events_from_runs(runs_dir: str | Path, game: str | None = None
                     ) -> dict[str, dict[str, list[tuple[str, str, float]]]]:
    """Collect (a, b, score_a) events from result.json, then match-*.json
    (skipping duplicate duel ids), bucketed by game and size_key(game_config)."""
    root, by_game, seen = Path(runs_dir), {}, set()

    def add(g: str, cfg: dict | None, a: str, b: str, sa: float, did: str | None) -> None:
        by_game.setdefault(g, {}).setdefault(size_key(cfg), []).append((a, b, sa))
        if did:
            seen.add(did)

    for path in root.rglob("result.json"):
        ev = json.loads(path.read_text())
        g = ev.get("game") or ""
        if game and g != game:
            continue
        players = ev.get("players") or {}
        w = ev.get("winner")
        if w not in {"a", "b", "draw"} or "a" not in players or "b" not in players:
            continue
        sa = {"a": 1.0, "b": 0.0, "draw": 0.5}[w]
        add(g, ev.get("game_config"), players["a"], players["b"], sa, ev.get("duel_id"))
    for path in root.rglob("match-*.json"):
        m = json.loads(path.read_text())
        g = m.get("game") or ""
        if game and g != game:
            continue
        names = m.get("players") or {}
        for d in m.get("duels") or []:
            if d.get("duel_id") in seen:
                continue
            if "score_a" not in d or "a" not in names or "b" not in names:
                continue
            add(g, m.get("game_config"), names["a"], names["b"], float(d["score_a"]), d.get("duel_id"))
    return by_game


def fit_runs(runs_dir: str | Path, game: str | None = None, n_boot: int = 400,
             seed: int = 0) -> dict[str, dict]:
    """{game: {size: {player: {rating, lo, hi, w, l, d}}}}."""
    return {g: {size: bootstrap(evs, n_boot=n_boot, seed=seed) for size, evs in buckets.items()}
            for g, buckets in events_from_runs(runs_dir, game).items()}
