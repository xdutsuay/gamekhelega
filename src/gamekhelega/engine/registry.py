"""Game discovery: built-in `gamekhelega.games.<name>` packages + entry points."""
from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import entry_points

from .game import PuzzleGame

ENTRY_POINT_GROUP = "gamekhelega.games"
_cache: dict[str, PuzzleGame] | None = None


def _discover() -> dict[str, PuzzleGame]:
    games: dict[str, PuzzleGame] = {}
    import gamekhelega.games as pkg

    for m in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"{pkg.__name__}.{m.name}")
        g = getattr(mod, "GAME", None)
        if isinstance(g, PuzzleGame):
            games[g.name] = g
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        g = ep.load()
        if isinstance(g, PuzzleGame):
            games[g.name] = g
    return games


def all_games() -> dict[str, PuzzleGame]:
    global _cache
    if _cache is None:
        _cache = _discover()
    return _cache


def get_game(name: str) -> PuzzleGame:
    games = all_games()
    if name not in games:
        raise KeyError(f"unknown game {name!r}; available: {sorted(games)}")
    return games[name]
