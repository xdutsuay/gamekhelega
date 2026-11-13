"""Keep the default test process in-process: spawning a pool (and JIT-compiling
numba in each child) would add seconds to every mock duel. Tests that want a
real process pool pass workers= explicitly.
"""
import os

os.environ.setdefault("GAMEKHELEGA_VALIDATION_WORKERS", "1")
