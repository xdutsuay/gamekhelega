import json

import numpy as np
import pytest

from gamekhelega.games.zip3d import GAME
from gamekhelega.games.zip3d.generator import (count, puzzle_from_positions, random_hamiltonian_path,
                                                serpentine)
from gamekhelega.games.zip3d.model import ZipPuzzle, adjacent, cell, edge
from gamekhelega.games.zip3d.solver_cpsat import cpsat_count
from gamekhelega.games.zip3d.solver_dfs import dfs_solve, parity_ok


def _is_ham(n, path):
    return sorted(path) == list(range(n ** 3)) and all(adjacent(n, a, b) for a, b in zip(path, path[1:]))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_paths_are_hamiltonian(n):
    assert _is_ham(n, serpentine(n))
    rng = np.random.default_rng(n)
    for _ in range(3):
        assert _is_ham(n, random_hamiltonian_path(n, rng))


def test_config_validation():
    assert GAME.parse_config({}).n == 4
    with pytest.raises(ValueError):
        GAME.parse_config({"n": 7})
    with pytest.raises(ValueError):
        GAME.parse_config({"n": 4, "colour": 1})


@pytest.mark.parametrize("n", [3, 4])
def test_generate_unique_and_solvers_agree(n):
    cfg = GAME.parse_config({"n": n})
    rng = np.random.default_rng(7)
    for _ in range(3 if n == 3 else 1):
        p, s = GAME.generate(cfg, "medium", rng, time_budget=60)
        assert GAME.validate_puzzle(cfg, p) == []
        assert GAME.check_solution(cfg, p, s).ok
        assert GAME.count_solutions(cfg, p, time_limit=60) == 1
        k, sols = cpsat_count(p, 2, time_limit=60)
        assert k == 1 and sols[0] == s
        assert GAME.solve(cfg, p, time_limit=60) == s
        assert 0 <= GAME.rate(cfg, p) <= 10


def test_generate_with_walls_3():
    cfg = GAME.parse_config({"n": 3, "walls": True})
    rng = np.random.default_rng(1)
    for _ in range(5):
        p, s = GAME.generate(cfg, "hard", rng)
        assert GAME.check_solution(cfg, p, s).ok and GAME.count_solutions(cfg, p) == 1
        assert len(p.walls) <= cfg.wall_cap()
        assert not ({edge(a, b) for a, b in zip(s, s[1:])} & p.walls)


def test_non_unique_detected():
    cfg = GAME.parse_config({"n": 3})
    path = serpentine(3)
    p = puzzle_from_positions(3, path, [0, 26])  # only start and end: many paths
    assert dfs_solve(p, limit=2).count == 2
    assert cpsat_count(p, 2, time_limit=30)[0] == 2
    assert GAME.count_solutions(cfg, p) == 2
    # 4^3 with just endpoints: DFS gives up early, CP-SAT must find a 2nd path
    p4 = puzzle_from_positions(4, serpentine(4), [0, 63])
    assert count(p4, 2, node_cap=100, time_limit=30)[0] == 2


def test_impossible_puzzles():
    # same colour endpoints on an even grid -> parity says no path
    p = ZipPuzzle(4, {cell(4, 0, 0, 0): 1, cell(4, 1, 1, 0): 2}, set())
    assert not parity_ok(p) and dfs_solve(p).count == 0
    assert GAME.count_solutions(GAME.parse_config({"n": 4}), p) == 0
    # a corner walled off on all three faces
    walls = {edge(0, 1), edge(0, 3), edge(0, 9)}
    p3 = ZipPuzzle(3, {0: 1, 26: 2}, walls)
    assert dfs_solve(p3).count == 0 and cpsat_count(p3, 2, time_limit=10)[0] == 0


def test_json_roundtrip_and_errors():
    cfg = GAME.parse_config({"n": 3, "walls": True})
    p, s = GAME.generate(cfg, "hard", np.random.default_rng(3))
    p.walls.add(edge(cell(3, 0, 0, 0), cell(3, 0, 0, 1)))
    j = json.loads(json.dumps(GAME.puzzle_to_json(cfg, p)))
    q = GAME.puzzle_from_json(cfg, j)
    assert q.waypoints == p.waypoints and q.walls == p.walls
    assert GAME.solution_from_json(cfg, GAME.solution_to_json(cfg, s)) == s
    # move-string form
    moves = []
    for a, b in zip(s, s[1:]):
        d = b - a
        moves.append({1: "+x", -1: "-x", 3: "+y", -3: "-y", 9: "+z", -9: "-z"}[d])
    sj = GAME.solution_to_json(cfg, s)["path"][0]
    assert GAME.solution_from_json(cfg, {"start": sj, "moves": ",".join(moves)}) == s
    # "-x" wall spelling is the same face as "+x" from the neighbour
    a = GAME.puzzle_from_json(cfg, {"waypoints": [[1, 1, 1, 1], [3, 3, 3, 2]], "walls": [[2, 1, 1, "-x"]]})
    assert a.walls == {edge(0, 1)}
    for bad in [{"waypoints": [[0, 1, 1, 1]]}, {"waypoints": [[1, 1, 1]]}, {"layers": []},
                {"waypoints": [[1, 1, 1, 1]], "walls": [[3, 1, 1, "+x"]]},
                {"waypoints": [[1, 1, 1, 1], [1, 1, 1, 2]]}]:
        with pytest.raises(ValueError):
            GAME.puzzle_from_json(cfg, bad)
    with pytest.raises(ValueError):
        GAME.solution_from_json(cfg, {"path": [[1, 1, 4]]})
    assert GAME.validate_puzzle(cfg, ZipPuzzle(3, {0: 1, 26: 3}, set()))  # gap in numbering
    assert GAME.validate_puzzle(cfg, ZipPuzzle(3, {0: 1}, set()))


def test_check_solution_errors_and_partial():
    cfg = GAME.parse_config({"n": 3})
    p, s = GAME.generate(cfg, "medium", np.random.default_rng(4))
    r = GAME.check_solution(cfg, p, s[:-1], reference=s)
    assert not r.ok and any("cells" in e for e in r.errors) and 0.9 < r.partial < 1
    rev = GAME.check_solution(cfg, p, s[::-1], reference=s)
    assert not rev.ok and any("start" in e for e in rev.errors)
    jump = s[:2] + s[3:] + [s[2]]
    assert any("adjacent" in e for e in GAME.check_solution(cfg, p, jump).errors)
    walled = ZipPuzzle(3, p.waypoints, {edge(s[5], s[6])})
    assert any("wall" in e for e in GAME.check_solution(cfg, walled, s).errors)


def test_human_moves():
    cfg = GAME.parse_config({"n": 3})
    p, s = GAME.generate(cfg, "medium", np.random.default_rng(5))
    st = GAME.new_state(cfg, p)
    names = {1: "+x", -1: "-x", 3: "+y", -3: "-y", 9: "+z", -9: "-z"}
    moves = [names[b - a] for a, b in zip(s, s[1:])]
    assert GAME.apply_move(cfg, st, GAME.parse_move(cfg, " ".join(moves[:5]))).startswith("ok")
    assert GAME.apply_move(cfg, st, GAME.parse_move(cfg, "undo")) == "undone"
    assert len(st.path) == 1
    for m in moves[:10]:
        GAME.apply_move(cfg, st, GAME.parse_move(cfg, m))
    x, y, z = GAME.solution_to_json(cfg, [s[4]])["path"][0]
    assert "cut back" in GAME.apply_move(cfg, st, GAME.parse_move(cfg, f"goto {x} {y} {z}"))
    assert st.path == s[:5]
    for c in s[5:]:
        x, y, z = GAME.solution_to_json(cfg, [c])["path"][0]
        GAME.apply_move(cfg, st, GAME.parse_move(cfg, f"goto {x} {y} {z}"))
    assert GAME.check_solution(cfg, p, GAME.state_to_solution(cfg, st)).ok
    assert GAME.apply_move(cfg, st, GAME.parse_move(cfg, "view 2")) == "viewing z=2"
    assert "27/27" in GAME.apply_move(cfg, st, GAME.parse_move(cfg, "check"))
    GAME.apply_move(cfg, st, GAME.parse_move(cfg, "reset"))
    assert len(st.path) == 1
    for bad in ["fly", "move +w", "goto 1 1", "goto 4 1 1", "view 9"]:
        with pytest.raises(ValueError):
            GAME.parse_move(cfg, bad)
    wp = ZipPuzzle(3, {0: 1, 26: 2}, {edge(0, 1)})
    st2 = GAME.new_state(cfg, wp)
    assert "wall" in GAME.apply_move(cfg, st2, GAME.parse_move(cfg, "+x"))
    assert "leaves" in GAME.apply_move(cfg, st2, GAME.parse_move(cfg, "-x"))


def test_prompts_and_schemas():
    cfg = GAME.parse_config({"n": 4})
    rules = GAME.prompt_rules(cfg)
    assert "64" in rules and "waypoint" in rules
    assert GAME.prompt_make_tips(cfg) and GAME.prompt_solve_tips(cfg)
    assert GAME.solution_json_schema(cfg)["properties"]["path"]["minItems"] == 64
    assert "waypoints" in GAME.puzzle_json_schema(cfg)["properties"]


def test_renderers():
    from rich.console import Console
    cfg = GAME.parse_config({"n": 4, "walls": True})
    p, s = GAME.generate(cfg, "hard", np.random.default_rng(6), time_budget=60)
    con = Console(record=True, width=200)
    con.print(GAME.render_terminal(cfg, p, s))
    out = con.export_text()
    assert "64/64" in out and "z=4" in out
    fig = GAME.render_3d(cfg, p, s)
    assert [d.name for d in fig.data] == ["cells", "waypoints", "path", "walls"]
    steps = fig.layout.sliders[0].steps
    assert [x.label for x in steps] == ["all", "exploded", "1", "2", "3", "4"]
    assert len(fig.layout.updatemenus) >= 1
    fig2 = GAME.render_3d(cfg, p)  # no solution: still renders
    assert len(fig2.data) == 4


def test_mock_duel_zip3d(tmp_path):
    from gamekhelega.engine.adapters.base import make_player
    from gamekhelega.engine.duel import DuelConfig, run_duel
    dc = DuelConfig(game="zip3d", game_config={"n": 3}, puzzles_per_side=2, make_time_s=30,
                    solve_time_s=30, out_dir=str(tmp_path), leaderboard=str(tmp_path / "lb.json"))
    a = make_player({"type": "mock", "id": "strong", "skill": 1.0, "difficulty": "hard", "seed": 1}, GAME)
    b = make_player({"type": "mock", "id": "weak", "skill": 0.0, "bad_make": 1.0, "seed": 2}, GAME)
    res = run_duel(GAME, a, b, dc)
    assert res["winner"] == "a"
    assert res["scores"]["b"]["n_correct"] == 0 and res["scores"]["a"]["n_correct"] == 0
    assert json.loads((tmp_path / "lb.json").read_text())["zip3d"]["strong"]["w"] == 1


def test_cli_zip3d(tmp_path):
    from typer.testing import CliRunner
    from gamekhelega.cli import app
    runner = CliRunner()
    f = tmp_path / "z.json"
    r = runner.invoke(app, ["generate", "--game", "zip3d", "-p", "n=3", "--count", "2", "--seed", "1",
                            "--out", str(f)])
    assert r.exit_code == 0, r.output
    assert runner.invoke(app, ["check", str(f)]).exit_code == 0
    assert runner.invoke(app, ["solve", str(f)]).exit_code == 0
    r = runner.invoke(app, ["view", str(f), "--html", str(tmp_path / "v.html")])
    assert r.exit_code == 0 and (tmp_path / "v.html").exists()
    assert "zip3d" in runner.invoke(app, ["games"]).output
