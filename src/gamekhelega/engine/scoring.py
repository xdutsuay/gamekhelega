"""Game-agnostic duel scoring.

Per side X (opponent Y), over the puzzles Y made that were accepted:
  solve_pts = sum_i [ W_SOLVE*c_i + W_PARTIAL*(1-c_i)*p_i ] + W_SPEED*n_correct*(1 - t_used/T_solve)
Over X's own submissions:
  make_pts  = sum_{accepted j} [ W_STUMP*(1-c'_j) + W_DIFF*d_j/10 ] - W_INVALID*n_rejected_or_missing
Walkover: each rejected/missing puzzle of Y gives X + W_WALKOVER (X couldn't be tested on it).
total = solve_pts + make_pts + walkover_pts
c_i correct (0/1), p_i partial in [0,1], d_j referee difficulty 0..10.
Tie-break: more correct solves > less solve time > fewer rejected puzzles > draw.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

W_SOLVE, W_PARTIAL, W_SPEED = 10.0, 3.0, 5.0
W_STUMP, W_DIFF = 6.0, 4.0
W_INVALID, W_WALKOVER = 15.0, 10.0


@dataclass
class SideScore:
    solve_pts: float = 0.0
    make_pts: float = 0.0
    walkover_pts: float = 0.0
    total: float = 0.0
    n_correct: int = 0
    n_attempted: int = 0
    solve_time_s: float = 0.0
    n_rejected: int = 0
    n_stumped: int = 0

    def to_json(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def score_side(solves: list[dict], own_made: list[dict], n_requested: int,
               opp_rejected: int, solve_time_limit: float) -> SideScore:
    """solves: [{"correct":bool,"partial":float,"time_s":float}] for opponent's accepted puzzles.
    own_made: [{"accepted":bool,"difficulty":float,"opponent_correct":bool|None}] own submissions."""
    s = SideScore()
    s.n_attempted = len(solves)
    s.n_correct = sum(1 for x in solves if x["correct"])
    s.solve_time_s = sum(x["time_s"] for x in solves)
    s.solve_pts = sum(W_SOLVE if x["correct"] else W_PARTIAL * x.get("partial", 0.0) for x in solves)
    if solve_time_limit > 0:
        s.solve_pts += W_SPEED * s.n_correct * max(0.0, 1 - s.solve_time_s / solve_time_limit)
    accepted = [m for m in own_made if m["accepted"]]
    s.n_rejected = n_requested - len(accepted)
    s.n_stumped = sum(1 for m in accepted if not m.get("opponent_correct"))
    s.make_pts = (sum(W_STUMP * (0 if m.get("opponent_correct") else 1) + W_DIFF * m["difficulty"] / 10
                      for m in accepted) - W_INVALID * s.n_rejected)
    s.walkover_pts = W_WALKOVER * opp_rejected
    s.total = s.solve_pts + s.make_pts + s.walkover_pts
    return s


def winner(a: SideScore, b: SideScore) -> str:
    """Returns "a", "b" or "draw"."""
    keys = [(a.total, b.total, 1), (a.n_correct, b.n_correct, 1),
            (a.solve_time_s, b.solve_time_s, -1), (a.n_rejected, b.n_rejected, -1)]
    for x, y, sign in keys:
        if abs(x - y) > 1e-9:
            return "a" if (x - y) * sign > 0 else "b"
    return "draw"
