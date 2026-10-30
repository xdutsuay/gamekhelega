"""Subprocess player: one process per request; Request JSON on stdin, reply JSON on stdout.

Lets anyone plug in a local model, a hand-written solver, or a wrapper around
any vendor SDK without touching this codebase.
"""
from __future__ import annotations

import json
import subprocess

from ..protocol import Request, extract_json
from .base import Player, Reply


class CommandPlayer(Player):
    def __init__(self, cmd: list[str] | str, id: str = "command", cwd: str | None = None,
                 env: dict | None = None):
        self.cmd = cmd if isinstance(cmd, list) else cmd.split()
        self.id, self.cwd, self.env = id, cwd, env

    def act(self, req: Request) -> Reply:
        try:
            p = subprocess.run(self.cmd, input=json.dumps(req.to_json()), capture_output=True,
                               text=True, timeout=req.time_limit_s, cwd=self.cwd, env=self.env)
        except subprocess.TimeoutExpired:
            return Reply(error="timeout")
        except OSError as e:
            return Reply(error=f"spawn: {e}")
        if p.returncode != 0:
            return Reply(error=f"exit {p.returncode}: {p.stderr[-500:]}", raw=p.stdout[:4000])
        try:
            return Reply(data=extract_json(p.stdout), raw=p.stdout[:4000])
        except ValueError as e:
            return Reply(error=f"parse: {e}", raw=p.stdout[:4000])
