"""The engine and game registry must import without numba, ortools, or python-sat."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

PROBE = r"""
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"numba", "ortools", "pysat"}:
            raise ImportError(f"blocked {name}")
        return None

sys.meta_path.insert(0, _Block())

import gamekhelega.engine.duel  # noqa: F401
import gamekhelega.engine.elo  # noqa: F401
import gamekhelega.engine.match  # noqa: F401
import gamekhelega.engine.protocol  # noqa: F401
import gamekhelega.engine.scoring  # noqa: F401
import gamekhelega.engine.validate  # noqa: F401
from gamekhelega.engine.adapters.mock import MockPlayer  # noqa: F401
from gamekhelega.engine.registry import all_games

games = all_games()
assert "sudoku3d" in games and "zip3d" in games, sorted(games)

try:
    from gamekhelega.games.sudoku3d.solver_numba import solve  # noqa: F401
except ImportError as e:
    assert "gamekhelega[solvers]" in str(e), e
else:
    raise SystemExit("expected ImportError from solver_numba")
print("ok")
"""


def test_engine_imports_with_solvers_blocked():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    assert "ok" in r.stdout
