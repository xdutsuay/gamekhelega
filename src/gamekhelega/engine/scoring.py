"""Game-agnostic duel scoring.

Per side X (opponent Y), over the puzzles Y made that were accepted:
  solve_pts = sum_i [ w.solve*c_i + w.partial*(1-c_i)*p_i ] + w.speed*n_correct*(1 - t_used/T_solve)
Over X's own submissions:
  make_pts  = sum_{accepted j} [ w.stump*(1 if c'_j is False else 0) + w.difficulty*d_j/10 ]
              - w.invalid*n_rejected_or_missing
Walkover: each rejected/missing puzzle of Y gives X + w.walkover (X couldn't be tested on it).
total = solve_pts + make_pts + walkover_pts
c_i correct (0/1), p_i partial in [0,1], d_j referee difficulty 0..10.
c'_j is True/False if the opponent attempted the puzzle, or None if they
never started it (ran out of time). None is not a stump: you only earn
stump points when the opponent actually tried and failed.
Tie-break: more correct solves > less solve time > fewer rejected puzzles > draw.

Weights are ScoringWeights (defaults below). Override any of them from the
duel JSON under "scoring". The defaults encode the policy in DESIGN §11:
solve beats stump, an invalid is worse than any valid puzzle, and speed is
only a bonus on top of being correct.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Named policy knobs for score_side. Defaults are the documented scale."""
    solve: float = 10.0       # points for a correct solution
    partial: float = 3.0      # max consolation for an incorrect attempt (× partial in [0,1])
    speed: float = 5.0        # bonus only when correct, scaled by unused solve-budget fraction
    stump: float = 6.0        # bonus when the opponent tried the puzzle and failed
    difficulty: float = 4.0   # max bonus for a hard valid puzzle (× d/10)
    invalid: float = 15.0     # penalty per own rejected or missing puzzle
    walkover: float = 10.0    # compensation per opponent puzzle you could not be tested on

    @classmethod
    def from_dict(cls, d: dict | None) -> "ScoringWeights":
        if not d:
            return cls()
        known = {k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


DEFAULT_WEIGHTS = ScoringWeights()
# Aliases so existing tests and call sites can keep using the old names.
W_SOLVE, W_PARTIAL, W_SPEED = DEFAULT_WEIGHTS.solve, DEFAULT_WEIGHTS.partial, DEFAULT_WEIGHTS.speed
W_STUMP, W_DIFF = DEFAULT_WEIGHTS.stump, DEFAULT_WEIGHTS.difficulty
W_INVALID, W_WALKOVER = DEFAULT_WEIGHTS.invalid, DEFAULT_WEIGHTS.walkover


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
               opp_rejected: int, solve_time_limit: float,
               weights: ScoringWeights | None = None) -> SideScore:
    """solves: [{"correct":bool|None,"partial":float,"time_s":float}] for opponent's accepted puzzles.
    own_made: [{"accepted":bool,"difficulty":float,"opponent_correct":bool|None}] own submissions.
    opponent_correct/correct is None when the side never started the puzzle."""
    w = weights or DEFAULT_WEIGHTS
    s = SideScore()
    attempted = [x for x in solves if x.get("correct") is not None]
    s.n_attempted = len(attempted)
    s.n_correct = sum(1 for x in attempted if x["correct"])
    s.solve_time_s = sum(x["time_s"] for x in attempted)
    s.solve_pts = sum(w.solve if x["correct"] else w.partial * x.get("partial", 0.0) for x in attempted)
    if solve_time_limit > 0:
        s.solve_pts += w.speed * s.n_correct * max(0.0, 1 - s.solve_time_s / solve_time_limit)
    accepted = [m for m in own_made if m["accepted"]]
    s.n_rejected = n_requested - len(accepted)
    s.n_stumped = sum(1 for m in accepted if m.get("opponent_correct") is False)
    s.make_pts = (sum(w.stump * (1 if m.get("opponent_correct") is False else 0)
                      + w.difficulty * m["difficulty"] / 10
                      for m in accepted) - w.invalid * s.n_rejected)
    s.walkover_pts = w.walkover * opp_rejected
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
