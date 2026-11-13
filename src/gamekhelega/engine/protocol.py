"""Game-agnostic request/response envelopes + prompt construction.

Every player (LLM, subprocess, human, mock) receives a `Request` and returns a
JSON-able dict. The referee measures wall-clock time itself.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .game import PuzzleGame

SYSTEM_PROMPT = (
    "You are a contestant in a timed puzzle duel refereed by a program. "
    "Reply with exactly ONE JSON object and nothing else: no prose, no markdown, "
    "no code fences. It must match the JSON Schema you are given. "
    "Malformed output forfeits the item. The referee checks everything exactly."
)


@dataclass
class Request:
    phase: str                    # "make" | "solve"
    game: str
    config: dict
    rules: str                    # game.prompt_rules(cfg)
    time_limit_s: float           # budget for THIS request (wall clock, referee-measured)
    count: int = 0                # make: how many puzzles to produce
    puzzle_id: str = ""           # solve: id of the puzzle
    puzzle: Any = None            # solve: canonical puzzle JSON
    tips: str = ""
    output_schema: dict = field(default_factory=dict)
    # Referee-set; not part of the wire format. Adapters should abort when set.
    cancel: Optional[Any] = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("cancel", None)
        return d


def make_schema(game: PuzzleGame, cfg: Any, count: int) -> dict:
    return {
        "type": "object",
        "required": ["puzzles"],
        "properties": {
            "puzzles": {
                "type": "array", "maxItems": count,
                "items": {
                    "type": "object", "required": ["puzzle", "solution"],
                    "properties": {
                        "puzzle": game.puzzle_json_schema(cfg),
                        "solution": game.solution_json_schema(cfg),
                    },
                },
            }
        },
    }


def solve_schema(game: PuzzleGame, cfg: Any) -> dict:
    return {
        "type": "object",
        "required": ["solution"],
        "properties": {"solution": {"anyOf": [game.solution_json_schema(cfg), {"type": "null"}]}},
    }


def make_request(game: PuzzleGame, cfg: Any, count: int, time_limit_s: float) -> Request:
    return Request("make", game.name, game.config_to_json(cfg), game.prompt_rules(cfg),
                   time_limit_s, count=count, tips=game.prompt_make_tips(cfg),
                   output_schema=make_schema(game, cfg, count))


def solve_request(game: PuzzleGame, cfg: Any, puzzle_id: str, puzzle_json: Any,
                  time_limit_s: float) -> Request:
    return Request("solve", game.name, game.config_to_json(cfg), game.prompt_rules(cfg),
                   time_limit_s, puzzle_id=puzzle_id, puzzle=puzzle_json,
                   tips=game.prompt_solve_tips(cfg), output_schema=solve_schema(game, cfg))


def user_prompt(req: Request) -> str:
    """The exact user message sent to chat models."""
    head = f"GAME: {req.game}\nCONFIG: {json.dumps(req.config)}\n\nRULES:\n{req.rules}\n"
    if req.phase == "make":
        task = (
            f"\nTASK (MAKE): Create {req.count} puzzles that are as HARD as possible for a strong "
            "opponent to solve, but each MUST have exactly one solution. For each puzzle also give "
            "its full solution. The referee rejects (and penalises you for) any puzzle that is "
            "malformed, whose solution you give is wrong, or that has 0 or 2+ solutions.\n"
            f"Time limit: {req.time_limit_s:.0f} s wall clock for this whole reply.\n"
        )
    else:
        task = (
            f"\nTASK (SOLVE): Solve puzzle {req.puzzle_id}. It has exactly one solution.\n"
            f"Time limit: {req.time_limit_s:.0f} s wall clock. If you cannot solve it, reply "
            '{"solution": null} (a wrong answer may earn partial credit, a null earns nothing).\n'
            f"\nPUZZLE:\n{json.dumps(req.puzzle)}\n"
        )
    tips = f"\nTIPS:\n{req.tips}\n" if req.tips else ""
    return head + task + tips + "\nOUTPUT JSON SCHEMA:\n" + json.dumps(req.output_schema) + "\n"


_FENCE = re.compile(r"```([^\n`]*)\n?(.*?)```", re.DOTALL)


def extract_json(text: str, required: list[str] | None = None, schema: dict | None = None) -> Any:
    """Pull a JSON object out of untrusted model text.

    Models often write `{example}` or a scratch object in prose before the
    real answer. Taking the first `{` is how you parse the wrong one.
    Instead we scan every candidate with JSONDecoder.raw_decode, prefer a
    fenced ```json block, and then take the last valid object that has the
    expected keys (from `required` or `schema["required"]`). If nothing
    matches the keys, we fall back to the last valid object.
    """
    t = text.strip()
    keys = list(required or (schema or {}).get("required") or [])
    json_fences, other_fences = [], []
    for m in _FENCE.finditer(t):
        lang, body = m.group(1).strip().lower(), m.group(2)
        (json_fences if lang in ("json", "jsonc") else other_fences).append(body)
    for pool in (json_fences, other_fences, [t]):
        picked = _pick_object(pool, keys, matching_only=True)
        if picked is not None:
            return picked
    for pool in (json_fences, other_fences, [t]):
        picked = _pick_object(pool, keys, matching_only=False)
        if picked is not None:
            return picked
    raise ValueError("no JSON object in reply")


def _iter_objects(text: str):
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        start = text.find("{", i)
        if start < 0:
            return
        try:
            obj, end = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(obj, dict):
            yield obj
        i = end


def _pick_object(sources: list[str], keys: list[str], matching_only: bool):
    last = None
    for src in sources:
        for obj in _iter_objects(src):
            if matching_only and keys and not all(k in obj for k in keys):
                continue
            last = obj
    return last
