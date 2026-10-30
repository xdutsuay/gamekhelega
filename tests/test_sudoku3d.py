import numpy as np
import pytest

from gamekhelega.games.sudoku3d import GAME
from gamekhelega.games.sudoku3d.construct import base_grid, random_transform
from gamekhelega.games.sudoku3d.rules import Rules, conflicts, is_complete_solution
from gamekhelega.games.sudoku3d.solver_numba import solve
from gamekhelega.games.sudoku3d.solver_sat import SatSolver

ALL = [(4, "layer"), (4, "symmetric"), (9, "layer"), (9, "symmetric")]


@pytest.mark.parametrize("n,variant", ALL)
def test_base_and_transforms_valid(n, variant):
    r = Rules(n, variant)
    g = base_grid(r)
    assert is_complete_solution(g, r)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert is_complete_solution(random_transform(g, r, rng), r)


def test_layer_base_is_not_symmetric():
    # proves the two variants differ (layer grid violates x/y-slice boxes)
    assert not is_complete_solution(base_grid(Rules(9, "layer")), Rules(9, "symmetric"))


def test_unit_counts():
    assert Rules(9, "layer").units.shape == (3 * 81 + 81, 9)
    assert Rules(9, "symmetric").units.shape == (3 * 81 + 3 * 81, 9)


@pytest.mark.parametrize("n,variant", [(4, "layer"), (4, "symmetric")])
def test_generate_unique_and_solvers_agree(n, variant):
    cfg = GAME.parse_config({"n": n, "variant": variant})
    rng = np.random.default_rng(3)
    sat = SatSolver(cfg)
    for _ in range(5):
        p, s = GAME.generate(cfg, "hard", rng)
        assert GAME.validate_puzzle(cfg, p) == []
        assert GAME.check_solution(cfg, p, s).ok
        r = solve(p, cfg, limit=2)
        assert r.count == 1 and (r.solution == s).all()
        k, sol = sat.count(p, 2)
        assert k == 1 and (sol == s).all()
        assert GAME.count_solutions(cfg, p) == 1


def test_9_medium_generation_unique():
    cfg = GAME.parse_config({"n": 9})
    p, s = GAME.generate(cfg, "medium", np.random.default_rng(0), time_budget=20)
    assert GAME.count_solutions(cfg, p, time_limit=20) == 1
    assert (GAME.solve(cfg, p) == s).all()


def test_non_unique_detected():
    cfg = GAME.parse_config({"n": 4})
    assert GAME.count_solutions(cfg, np.zeros(cfg.shape, np.int8)) == 2


def test_json_roundtrip_and_errors():
    cfg = GAME.parse_config({"n": 9})
    p, s = GAME.generate(cfg, "easy", np.random.default_rng(1), time_budget=10)
    assert (GAME.puzzle_from_json(cfg, GAME.puzzle_to_json(cfg, p)) == p).all()
    assert (GAME.solution_from_json(cfg, GAME.solution_to_json(cfg, s)) == s).all()
    with pytest.raises(ValueError):
        GAME.solution_from_json(cfg, GAME.puzzle_to_json(cfg, p))  # has empties
    with pytest.raises(ValueError):
        GAME.puzzle_from_json(cfg, {"layers": [["123"]]})


def test_check_solution_partial_and_conflicts():
    cfg = GAME.parse_config({"n": 4})
    p, s = GAME.generate(cfg, "medium", np.random.default_rng(2))
    bad = s.copy()
    free = np.argwhere(p == 0)
    (z1, y1, x1), (z2, y2, x2) = free[0], free[1]
    bad[z1, y1, x1], bad[z2, y2, x2] = 0, 0
    r = GAME.check_solution(cfg, p, bad, reference=s)
    assert not r.ok and 0 < r.partial < 1
    assert conflicts(np.ones(cfg.shape, np.int8), cfg).all()


def test_human_moves():
    cfg = GAME.parse_config({"n": 4})
    p, s = GAME.generate(cfg, "medium", np.random.default_rng(4))
    st = GAME.new_state(cfg, p)
    for z, y, x in np.argwhere(p == 0):
        GAME.apply_move(cfg, st, GAME.parse_move(cfg, f"{x+1} {y+1} {z+1} {s[z, y, x]}"))
    assert GAME.check_solution(cfg, p, GAME.state_to_solution(cfg, st)).ok
    z, y, x = np.argwhere(p > 0)[0]
    assert "locked" in GAME.apply_move(cfg, st, GAME.parse_move(cfg, f"clear {x+1} {y+1} {z+1}"))
    assert GAME.apply_move(cfg, st, GAME.parse_move(cfg, "view x 2")).startswith("viewing")
    assert "empty" in GAME.apply_move(cfg, st, GAME.parse_move(cfg, "check"))
    with pytest.raises(ValueError):
        GAME.parse_move(cfg, "set 9 9 9 9")


def test_renderers():
    from rich.console import Console
    cfg = GAME.parse_config({"n": 4})
    p, s = GAME.generate(cfg, "medium", np.random.default_rng(5))
    con = Console(record=True, width=200)
    con.print(GAME.render_terminal(cfg, p, s))
    assert "givens" in con.export_text()
    fig = GAME.render_3d(cfg, p, s)
    assert len(fig.frames) == 0  # views switch via slider/dropdown "update" calls, no frames
    assert len(fig.layout.sliders[0].steps) == 4 + 2  # all, exploded, 1..n
    assert [d.name for d in fig.data] == ["given", "filled", "empty"]


def test_render_3d_viewer_n9(tmp_path):
    cfg = GAME.parse_config({"n": 9})
    sol = base_grid(cfg)
    puz = sol.copy()
    puz.ravel()[np.random.default_rng(1).permutation(puz.size)[:600]] = 0
    fig = GAME.render_3d(cfg, puz, sol)
    steps = fig.layout.sliders[0].steps
    assert len(steps) >= 10 and [s.label for s in steps[:3]] == ["all", "exploded", "1"]
    menus = {m.name: m for m in fig.layout.updatemenus}
    assert [b.label for b in menus["slice axis"].buttons] == ["z-layers", "y-slices", "x-slices"]
    assert "filled toggle" in menus
    # every cell appears exactly once in the "all" view, split by status
    given, filled, empty = fig.data
    assert len(given.x) == int((puz > 0).sum()) and len(filled.x) == 600 and len(empty.x) == 0
    assert "given" in given.hovertemplate and "filled" in filled.hovertemplate
    # one slice per frame, 1-based coordinates in hover data
    x_steps = menus["slice axis"].buttons[2].args[1]["sliders"][0]["steps"]
    upd = next(s for s in x_steps if s["label"] == "3")["args"][0]
    assert sum(len(xs) for xs in upd["x"]) == 81
    assert all(c[0] == 3 for cd in upd["customdata"] for c in cd)
    exploded = next(s for s in x_steps if s["label"] == "exploded")["args"][1]
    assert exploded["scene.xaxis.ticktext"] == [str(i) for i in range(1, 10)]
    assert exploded["scene.xaxis.tickvals"][1] > 2  # slices spread apart
    assert list(fig.layout.scene.xaxis.ticktext) == [str(i) for i in range(1, 10)]
    # puzzle-only view: empties shown, no filled toggle
    fig2 = GAME.render_3d(cfg, puz)
    assert len(fig2.data[2].x) == 600 and len(fig2.layout.updatemenus) == 1
    out = tmp_path / "cube.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    assert out.stat().st_size < 5 * 1024 * 1024
