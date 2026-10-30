"""Offline player backed by the game's own generator/solver (no API key needed).

skill in [0,1]: probability of solving a puzzle correctly; otherwise returns the
correct solution with a few cells/steps corrupted (to exercise partial credit).
`bad_make` in [0,1]: probability each made puzzle is deliberately broken.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from ..protocol import Request
from .base import Player, Reply


class MockPlayer(Player):
    def __init__(self, game, id: str = "mock", skill: float = 1.0, bad_make: float = 0.0,
                 difficulty: str = "medium", seed: int = 0, make_time_budget: float = 5.0):
        self.game, self.id, self.skill, self.bad_make = game, id, skill, bad_make
        self.difficulty, self.make_time_budget = difficulty, make_time_budget
        self.rng = np.random.default_rng(seed)

    def act(self, req: Request) -> Reply:
        g = self.game
        cfg = g.parse_config(req.config)
        if req.phase == "make":
            items = []
            per = min(self.make_time_budget, req.time_limit_s / max(1, req.count) * 0.8)
            for _ in range(req.count):
                p, s = g.generate(cfg, self.difficulty, self.rng, time_budget=per)
                item = {"puzzle": g.puzzle_to_json(cfg, p), "solution": g.solution_to_json(cfg, s)}
                if self.rng.random() < self.bad_make:
                    item["solution"] = _corrupt(item["solution"], self.rng)
                items.append(item)
            data = {"puzzles": items}
        else:
            p = g.puzzle_from_json(cfg, req.puzzle)
            s = g.solve(cfg, p, time_limit=req.time_limit_s)
            sj = None if s is None else g.solution_to_json(cfg, s)
            if sj is not None and self.rng.random() >= self.skill:
                sj = _corrupt(sj, self.rng)
            data = {"solution": sj}
        return Reply(data=data, raw=json.dumps(data)[:2000])


def _corrupt(obj, rng):
    """Game-agnostic corruption: swap two characters in one string leaf, or bump an int."""
    obj = copy.deepcopy(obj)

    def leaves(o, path=()):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from leaves(v, path + (k,))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from leaves(v, path + (i,))
        else:
            yield path, o

    ls = [(p, v) for p, v in leaves(obj) if isinstance(v, (str, int)) and not isinstance(v, bool)]
    if not ls:
        return obj
    path, v = ls[int(rng.integers(len(ls)))]
    if isinstance(v, str) and len(v) >= 2:
        i, j = rng.choice(len(v), 2, replace=False)
        lst = list(v); lst[i], lst[j] = lst[j], lst[i]; nv = "".join(lst)
        if nv == v:
            nv = v[::-1]
    elif isinstance(v, int):
        nv = v + 1
    else:
        return obj
    tgt = obj
    for k in path[:-1]:
        tgt = tgt[k]
    tgt[path[-1]] = nv
    return obj
