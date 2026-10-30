"""Duel referee: MAKE phase (both sides in parallel) -> validation -> SOLVE phase
(each side solves the opponent's accepted puzzles in order under one shared budget)
-> scoring -> results JSON + JSONL replay + optional Elo update.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import elo
from .adapters.base import Player, Reply
from .game import PuzzleGame
from .protocol import make_request, solve_request
from .scoring import score_side, winner
from .validate import check_submission


@dataclass
class DuelConfig:
    game: str
    game_config: dict = field(default_factory=dict)
    puzzles_per_side: int = 3
    make_time_s: float = 600.0
    solve_time_s: float = 900.0
    uniqueness_time_s: float = 180.0  # 9^3 layer 105-clue puzzle: cold pysat proof took ~90 s (research/bench_layer_hard_*.log)
    min_solve_request_s: float = 5.0   # don't send a solve request with less time than this
    seed: int = 0
    out_dir: str = "runs"
    leaderboard: str | None = "runs/leaderboard.json"

    @classmethod
    def from_dict(cls, d: dict) -> "DuelConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class Replay:
    def __init__(self, path: Path):
        self.path, self.t0 = path, time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w")

    def log(self, event: str, **kw: Any) -> None:
        self.f.write(json.dumps({"t": round(time.monotonic() - self.t0, 3), "event": event, **kw}) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _timed(pool, player: Player, req) -> tuple[Reply, float]:
    """Run player.act in its own daemon thread with a hard wall-clock limit.
    Late = error 'timeout' (the straggler thread is abandoned, never awaited)."""
    box: dict = {}

    def run() -> None:
        try:
            box["rep"] = player.act(req)
        except Exception as e:  # adapter bug
            box["rep"] = Reply(error=f"adapter: {type(e).__name__}: {e}")

    t0 = time.monotonic()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(req.time_limit_s + 1.0)  # 1 s grace for transport
    rep = box.get("rep") or Reply(error="timeout")
    dt = time.monotonic() - t0
    if dt > req.time_limit_s + 1.0:
        rep = Reply(error="timeout", raw=rep.raw, usage=rep.usage)
    return rep, dt


def run_duel(game: PuzzleGame, a: Player, b: Player, dc: DuelConfig) -> dict:
    cfg = game.parse_config(dc.game_config)
    duel_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    out = Path(dc.out_dir) / duel_id
    replay = Replay(out / "replay.jsonl")
    sides = {"a": a, "b": b}
    replay.log("start", duel_id=duel_id, game=game.name, config=game.config_to_json(cfg),
               players={k: p.id for k, p in sides.items()}, duel=dc.__dict__)
    pool = cf.ThreadPoolExecutor(max_workers=4)

    # ---- MAKE (parallel) ------------------------------------------------
    made: dict[str, list[dict]] = {}
    usage: dict[str, dict] = {"a": {}, "b": {}}
    futs = {k: pool.submit(_timed, pool, p, make_request(game, cfg, dc.puzzles_per_side, dc.make_time_s))
            for k, p in sides.items()}
    for k, f in futs.items():
        rep, dt = f.result()
        _add_usage(usage[k], rep.usage)
        items = []
        if rep.data and isinstance(rep.data.get("puzzles"), list):
            items = rep.data["puzzles"][: dc.puzzles_per_side]
        replay.log("make_reply", side=k, time_s=round(dt, 3), error=rep.error, n_items=len(items))
        verdicts = []
        for i, it in enumerate(items):
            v = check_submission(game, cfg, it if isinstance(it, dict) else {}, dc.uniqueness_time_s)
            pid = f"{k}{i + 1}"
            rec = {"id": pid, "maker": k, "accepted": v.accepted, "reason": v.reason,
                   "errors": v.errors[:5], "difficulty": v.difficulty,
                   "puzzle": game.puzzle_to_json(cfg, v.puzzle) if v.puzzle is not None else None,
                   "solution": game.solution_to_json(cfg, v.solution) if v.solution is not None else None}
            replay.log("verdict", **{x: rec[x] for x in ("id", "maker", "accepted", "reason", "errors", "difficulty")})
            verdicts.append(rec)
        made[k] = verdicts

    # ---- SOLVE (both sides in parallel, each sequential over its puzzles) --
    def solve_side(k: str) -> list[dict]:
        opp = "b" if k == "a" else "a"
        player = sides[k]
        results, spent = [], 0.0
        for rec in made[opp]:
            if not rec["accepted"]:
                continue
            left = dc.solve_time_s - spent
            r = {"id": rec["id"], "correct": False, "partial": 0.0, "time_s": 0.0, "error": ""}
            if left < dc.min_solve_request_s:
                r["error"] = "no time left"
                results.append(r)
                continue
            req = solve_request(game, cfg, rec["id"], rec["puzzle"], left)
            rep, dt = _timed(pool, player, req)
            _add_usage(usage[k], rep.usage)
            spent += dt
            r["time_s"] = round(dt, 3)
            r["error"] = rep.error
            sol_json = rep.data.get("solution") if isinstance(rep.data, dict) else None
            if sol_json is not None:
                try:
                    puz = game.puzzle_from_json(cfg, rec["puzzle"])
                    ref = game.solution_from_json(cfg, rec["solution"])
                    chk = game.check_solution(cfg, puz, game.solution_from_json(cfg, sol_json), ref)
                    r["correct"], r["partial"] = chk.ok, (1.0 if chk.ok else chk.partial)
                    r["errors"] = chk.errors[:5]
                except (ValueError, TypeError, KeyError) as e:
                    r["error"] = f"malformed solution: {e}"
            replay.log("solve", side=k, solution=sol_json, **{x: r[x] for x in ("id", "correct", "partial", "time_s", "error")})
            results.append(r)
        return results

    sf = {k: pool.submit(solve_side, k) for k in sides}
    solves = {k: f.result() for k, f in sf.items()}
    pool.shutdown(wait=False, cancel_futures=True)

    # ---- SCORE ----------------------------------------------------------
    scores = {}
    for k in sides:
        opp = "b" if k == "a" else "a"
        opp_correct = {r["id"]: r["correct"] for r in solves[opp]}
        own = [{"accepted": m["accepted"], "difficulty": m["difficulty"],
                "opponent_correct": opp_correct.get(m["id"])} for m in made[k]]
        opp_rej = dc.puzzles_per_side - sum(1 for m in made[opp] if m["accepted"])
        scores[k] = score_side(solves[k], own, dc.puzzles_per_side, opp_rej, dc.solve_time_s)
    w = winner(scores["a"], scores["b"])
    result = {
        "duel_id": duel_id, "game": game.name, "game_config": game.config_to_json(cfg),
        "duel_config": dc.__dict__, "players": {k: p.id for k, p in sides.items()},
        "puzzles": made, "solves": solves, "usage": usage,
        "scores": {k: s.to_json() for k, s in scores.items()},
        "winner": w, "winner_id": sides[w].id if w in sides else None,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    replay.log("end", winner=w, scores=result["scores"])
    replay.close()
    if dc.leaderboard:
        result["leaderboard"] = elo.record(dc.leaderboard, game.name, a.id, b.id, w)
    result["out_dir"] = str(out)
    return result


def _add_usage(acc: dict, u: dict) -> None:
    for k, v in (u or {}).items():
        if isinstance(v, (int, float)):
            acc[k] = acc.get(k, 0) + v
