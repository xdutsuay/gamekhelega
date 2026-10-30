"""Game-agnostic puzzle files:
{"format": "gamekhelega/v1", "game": "<name>", "config": {...},
 "items": [{"id": "p1", "puzzle": <game json>, "solution": <game json|null>, "meta": {...}}]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORMAT = "gamekhelega/v1"


def dump_pack(path: str | Path, game: str, config: dict, items: list[dict]) -> None:
    Path(path).write_text(json.dumps({"format": FORMAT, "game": game, "config": config,
                                      "items": items}, indent=1))


def load_pack(path: str | Path) -> dict[str, Any]:
    d = json.loads(Path(path).read_text())
    if d.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    return d


def parse_params(params: list[str] | None, json_cfg: str | None = None) -> dict:
    """--config '{"n":4}' and/or repeated --param n=4 (values parsed as JSON when possible)."""
    d: dict = json.loads(json_cfg) if json_cfg else {}
    for p in params or []:
        k, _, v = p.partition("=")
        try:
            d[k] = json.loads(v)
        except json.JSONDecodeError:
            d[k] = v
    return d
