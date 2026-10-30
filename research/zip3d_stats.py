"""Measure zip3d generation: time, waypoints, walls, rating, final uniqueness re-check.

usage: python research/zip3d_stats.py N SAMPLES DIFFICULTY BUDGET_S [walls]
Every number in research/zip3d_gen.log comes from this script.
"""
import statistics as st
import sys
import time

import numpy as np

from gamekhelega.games.zip3d import generator as gen
from gamekhelega.games.zip3d.model import ZipConfig

n, samples, diff, budget = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
walls = len(sys.argv) > 5 and sys.argv[5] == "walls"
cfg = ZipConfig(n=n, walls=walls)
times, ks, ws, scores, uniq = [], [], [], [], []
for seed in range(samples):
    t = time.monotonic()
    p, path = gen.generate(cfg, np.random.default_rng(seed), diff, time_budget=budget)
    dt = time.monotonic() - t
    t2 = time.monotonic()
    k, _ = gen.count(p, 2, time_limit=60)
    dv = time.monotonic() - t2
    r = gen.rate(p)
    times.append(dt); ks.append(p.K); ws.append(len(p.walls)); scores.append(r.score); uniq.append(k)
    print(f"n={n} {diff} walls={walls} seed={seed}: gen {dt:.2f}s K={p.K} ({p.K / n**3:.0%}) "
          f"walls={len(p.walls)} unique_recheck={k} ({dv:.2f}s) rate={r.score} dfs_nodes={r.nodes}", flush=True)
print(f"SUMMARY n={n} {diff} walls={walls} budget={budget}s samples={samples}: "
      f"gen median {st.median(times):.2f}s max {max(times):.2f}s | K median {st.median(ks)} "
      f"range {min(ks)}-{max(ks)} | walls mean {st.mean(ws):.1f} | rate median {st.median(scores)} | "
      f"unique {sum(1 for u in uniq if u == 1)}/{samples}", flush=True)
