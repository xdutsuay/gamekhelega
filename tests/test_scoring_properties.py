"""Property tests for the scoring invariants documented in scoring.py and DESIGN §11.

These turn the docstring into a spec: if someone changes a weight or the
tie-break order, these fail before a silent change in who wins a duel.
"""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from gamekhelega.engine.scoring import (
    W_DIFF, W_INVALID, W_STUMP, W_WALKOVER, SideScore, score_side, winner,
)


def test_invalid_costs_more_than_a_stump_is_worth():
    """Even a max-difficulty stump (+6 +4) is worth less than the -15 invalid penalty."""
    assert W_INVALID > W_STUMP + W_DIFF


@given(difficulty=st.floats(0, 10, allow_nan=False, allow_infinity=False))
def test_one_invalid_is_worse_than_one_stump(difficulty):
    stump = score_side([], [{"accepted": True, "difficulty": difficulty, "opponent_correct": False}],
                       1, 0, 100)
    invalid = score_side([], [{"accepted": False, "difficulty": 0}], 1, 0, 100)
    assert stump.make_pts == pytest.approx(W_STUMP + W_DIFF * difficulty / 10)
    assert invalid.make_pts == pytest.approx(-W_INVALID)
    assert invalid.make_pts < stump.make_pts


@given(
    opp_rejected=st.integers(0, 20),
    n_requested=st.integers(0, 10),
    n_solves=st.integers(0, 8),
    correct=st.booleans(),
    time_s=st.floats(0, 50, allow_nan=False, allow_infinity=False),
)
def test_walkover_points_depend_only_on_opponent_rejections(opp_rejected, n_requested, n_solves,
                                                            correct, time_s):
    """A side that could not be tested on k rejected opponent puzzles gets k * W_WALKOVER,
    regardless of how its own solves went."""
    solves = [{"correct": correct, "partial": 1.0 if correct else 0.0, "time_s": time_s}] * n_solves
    s = score_side(solves, [], n_requested, opp_rejected, 100)
    assert s.walkover_pts == pytest.approx(W_WALKOVER * opp_rejected)


@given(
    total_a=st.floats(-50, 50, allow_nan=False, allow_infinity=False),
    total_b=st.floats(-50, 50, allow_nan=False, allow_infinity=False),
    n_correct_a=st.integers(0, 10),
    n_correct_b=st.integers(0, 10),
    time_a=st.floats(0, 500, allow_nan=False, allow_infinity=False),
    time_b=st.floats(0, 500, allow_nan=False, allow_infinity=False),
    rej_a=st.integers(0, 10),
    rej_b=st.integers(0, 10),
)
def test_tie_break_order(total_a, total_b, n_correct_a, n_correct_b, time_a, time_b, rej_a, rej_b):
    """Winner: higher total, then more correct solves, then less solve time, then
    fewer rejected puzzles, then a draw. An earlier key always beats a later one."""
    a = SideScore(total=total_a, n_correct=n_correct_a, solve_time_s=time_a, n_rejected=rej_a)
    b = SideScore(total=total_b, n_correct=n_correct_b, solve_time_s=time_b, n_rejected=rej_b)
    w = winner(a, b)
    if abs(total_a - total_b) > 1e-9:
        assert w == ("a" if total_a > total_b else "b")
        return
    if n_correct_a != n_correct_b:
        assert w == ("a" if n_correct_a > n_correct_b else "b")
        return
    if abs(time_a - time_b) > 1e-9:
        assert w == ("a" if time_a < time_b else "b")
        return
    if rej_a != rej_b:
        assert w == ("a" if rej_a < rej_b else "b")
        return
    assert w == "draw"
