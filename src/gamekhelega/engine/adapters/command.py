"""Subprocess player: one process per request; Request JSON on stdin, reply JSON on stdout.

Lets anyone plug in a local model, a hand-written solver, or a wrapper around
any vendor SDK without touching this codebase.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time

from ..protocol import Request, extract_json
from .base import Player, Reply


class CommandPlayer(Player):
    def __init__(self, cmd: list[str] | str, id: str = "command", cwd: str | None = None,
                 env: dict | None = None):
        self.cmd = cmd if isinstance(cmd, list) else cmd.split()
        self.id, self.cwd, self.env = id, cwd, env

    def act(self, req: Request) -> Reply:
        kwargs: dict = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=self.cwd, env=self.env)
        if os.name == "posix":
            kwargs["start_new_session"] = True  # so we can kill the whole process group
        try:
            proc = subprocess.Popen(self.cmd, **kwargs)
        except OSError as e:
            return Reply(error=f"spawn: {e}")

        out_box, err_box = [], []

        def drain(src, box):
            try:
                box.append(src.read())
            except Exception:
                pass

        readers = [threading.Thread(target=drain, args=(proc.stdout, out_box), daemon=True),
                   threading.Thread(target=drain, args=(proc.stderr, err_box), daemon=True)]
        for t in readers:
            t.start()
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.write(json.dumps(req.to_json()))
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            deadline = time.monotonic() + req.time_limit_s
            while proc.poll() is None:
                if req.cancel is not None and req.cancel.is_set():
                    _kill_group(proc)
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    _kill_group(proc)
                    break
                try:
                    proc.wait(timeout=min(0.05, left))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                _kill_group(proc)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            for t in readers:
                t.join(1)

        stdout, stderr = "".join(out_box), "".join(err_box)
        if req.cancel is not None and req.cancel.is_set():
            return Reply(error="timeout", raw=stdout[:4000])
        if proc.returncode is None or (proc.returncode != 0 and time.monotonic() >= deadline):
            # killed because the wall-clock budget ran out
            if proc.returncode is None or proc.returncode < 0:
                return Reply(error="timeout", raw=stdout[:4000])
        if proc.returncode != 0:
            return Reply(error=f"exit {proc.returncode}: {stderr[-500:]}", raw=stdout[:4000])
        try:
            return Reply(data=extract_json(stdout, schema=req.output_schema), raw=stdout[:4000])
        except ValueError as e:
            return Reply(error=f"parse: {e}", raw=stdout[:4000])


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child and anything it spawned in the same process group."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix" and proc.pid:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass
