"""Per-game Elo leaderboard stored as JSON: {game: {player_id: {"elo":..,"w":..,"l":..,"d":..}}}.

The file is the result of a locked read-modify-write: writers take an
O_CREAT|O_EXCL lockfile, write a sibling temp file, then os.replace it into
place. Two duels finishing at once then cannot lose an update, and a crash
mid-write cannot leave a half-written JSON file.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

K, BASE = 32.0, 1500.0
_LOCK_TIMEOUT_S = 10.0
_LOCK_STALE_S = 30.0


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update(ra: float, rb: float, score_a: float, k: float = K) -> tuple[float, float]:
    ea = expected(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


@contextmanager
def _lock(path: Path, timeout: float = _LOCK_TIMEOUT_S, stale_s: float = _LOCK_STALE_S):
    """Cross-platform exclusive lock: create path.lock with O_EXCL, or wait.

    If a holder crashes, the lockfile is treated as stale after stale_s and
    stolen. No extra dependencies (fcntl is POSIX-only; this works on Windows too).
    """
    lock = path.with_name(path.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_s:
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {lock}")
            time.sleep(0.01)
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def apply(data: dict, game: str, a: str, b: str, result: str) -> dict:
    """Apply one duel to an in-memory leaderboard. result in {"a","b","draw"}."""
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
    return board


def record(path: str | os.PathLike, game: str, a: str, b: str, result: str) -> dict:
    """result in {"a","b","draw"}. Returns the updated board for `game`."""
    p = Path(path)
    with _lock(p):
        data = json.loads(p.read_text()) if p.exists() else {}
        board = apply(data, game, a, b, result)
        _atomic_write(p, json.dumps(data, indent=2))
        return board


def rebuild(runs_dir: str | os.PathLike, out: str | os.PathLike,
            game: str | None = None) -> dict:
    """Recompute a leaderboard from result.json files, oldest duel_id first.

    The run folders are the event log; the leaderboard is a view that can
    always be rebuilt from them. Existing `out` is replaced, not appended to.
    """
    events = []
    for path in Path(runs_dir).rglob("result.json"):
        ev = json.loads(path.read_text())
        if game and ev.get("game") != game:
            continue
        players = ev.get("players") or {}
        if ev.get("winner") not in {"a", "b", "draw"} or "a" not in players or "b" not in players:
            continue
        events.append(ev)
    events.sort(key=lambda ev: ev.get("duel_id", ""))
    data: dict = {}
    for ev in events:
        apply(data, ev["game"], ev["players"]["a"], ev["players"]["b"], ev["winner"])
    outp = Path(out)
    with _lock(outp):
        _atomic_write(outp, json.dumps(data, indent=2))
    return data
