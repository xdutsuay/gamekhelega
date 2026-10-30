"""Player adapter interface + factory."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..protocol import Request


@dataclass
class Reply:
    data: Any = None              # parsed JSON object (None on failure)
    raw: str = ""                 # raw text for the replay log (truncated by referee)
    error: str = ""               # transport/parse error, "" if ok
    usage: dict = field(default_factory=dict)  # e.g. {"prompt_tokens":..,"completion_tokens":..}


class Player(ABC):
    """One duel participant. `id` is used for Elo (e.g. "openai:gpt-4o-mini")."""

    id: str = "player"

    @abstractmethod
    def act(self, req: Request) -> Reply:
        """Blocking call. Must try to return within req.time_limit_s; the referee
        enforces the limit regardless (late replies are discarded)."""

    def close(self) -> None:  # optional cleanup
        pass


def make_player(spec: dict, game=None) -> Player:
    """spec examples:
    {"type": "mock", "id": "mock-a", "skill": 1.0, "seed": 1}
    {"type": "openai", "id": "gpt-4o-mini", "model": "gpt-4o-mini",
     "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"}
    {"type": "command", "id": "my-bot", "cmd": ["python", "bot.py"]}
    {"type": "human", "id": "alice"}
    """
    t = spec.get("type")
    if t == "mock":
        from .mock import MockPlayer
        return MockPlayer(game=game, **{k: v for k, v in spec.items() if k != "type"})
    if t == "openai":
        from .openai_compat import OpenAICompatPlayer
        return OpenAICompatPlayer(**{k: v for k, v in spec.items() if k != "type"})
    if t == "command":
        from .command import CommandPlayer
        return CommandPlayer(**{k: v for k, v in spec.items() if k != "type"})
    if t == "human":
        from .human import HumanPlayer
        return HumanPlayer(game=game, **{k: v for k, v in spec.items() if k != "type"})
    raise ValueError(f"unknown player type {t!r}")
