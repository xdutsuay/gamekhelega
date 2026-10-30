import json

import numpy as np
import pytest

from gamekhelega.engine import elo
from gamekhelega.engine.adapters.base import make_player
from gamekhelega.engine.adapters.command import CommandPlayer
from gamekhelega.engine.duel import DuelConfig, run_duel
from gamekhelega.engine.game import PuzzleGame
from gamekhelega.engine.protocol import extract_json, make_request, solve_request, user_prompt
from gamekhelega.engine.registry import all_games, get_game
from gamekhelega.engine.scoring import SideScore, score_side, winner
from gamekhelega.engine.validate import check_submission


def test_registry_discovers_games():
    g = all_games()
    assert "sudoku3d" in g and "zip3d" in g
    assert all(isinstance(x, PuzzleGame) for x in g.values())
    with pytest.raises(KeyError):
        get_game("nope")


def test_engine_has_no_game_specific_code():
    import pathlib
    root = pathlib.Path(__file__).parents[1] / "src/gamekhelega/engine"
    for f in root.rglob("*.py"):
        txt = f.read_text().lower()
        assert "sudoku" not in txt and "zip3d" not in txt, f


def test_check_submission_reasons():
    g = get_game("sudoku3d")
    cfg = g.parse_config({"n": 4})
    p, s = g.generate(cfg, "hard", np.random.default_rng(0))
    ok = {"puzzle": g.puzzle_to_json(cfg, p), "solution": g.solution_to_json(cfg, s)}
    assert check_submission(g, cfg, ok).accepted
    assert check_submission(g, cfg, {"puzzle": "x"}).reason == "malformed"
    empty = np.zeros(cfg.shape, np.int8)
    v = check_submission(g, cfg, {"puzzle": g.puzzle_to_json(cfg, empty), "solution": ok["solution"]})
    assert v.reason == "not_unique"
    wrong = s.copy(); wrong[0, 0, :] = wrong[0, 0, ::-1]
    v = check_submission(g, cfg, {"puzzle": ok["puzzle"], "solution": g.solution_to_json(cfg, wrong)})
    assert v.reason == "bad_solution"


def test_prompts_and_json_extraction():
    g = get_game("sudoku3d")
    cfg = g.parse_config({"n": 9})
    mk = user_prompt(make_request(g, cfg, 3, 600))
    assert "TASK (MAKE)" in mk and "exactly one solution" in mk and '"layers"' in mk
    sv = user_prompt(solve_request(g, cfg, "a1", {"layers": []}, 60))
    assert "TASK (SOLVE)" in sv and "a1" in sv
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"solution": null} hope it helps') == {"solution": None}


def test_scoring_and_winner():
    a = score_side([{"correct": True, "partial": 1, "time_s": 10}], [{"accepted": True, "difficulty": 5,
                   "opponent_correct": False}], 1, 0, 100)
    assert a.solve_pts == pytest.approx(10 + 5 * 0.9) and a.make_pts == pytest.approx(6 + 2)
    b = score_side([], [{"accepted": False, "difficulty": 0}], 1, 0, 100)
    assert b.make_pts == -15
    assert winner(a, b) == "a"
    assert winner(SideScore(total=1, n_correct=1, solve_time_s=5), SideScore(total=1, n_correct=1, solve_time_s=9)) == "a"


def test_elo(tmp_path):
    board = elo.record(tmp_path / "lb.json", "g", "p", "q", "a")
    assert board["p"]["elo"] == 1516.0 and board["q"]["elo"] == 1484.0
    board = elo.record(tmp_path / "lb.json", "g", "p", "q", "draw")
    assert board["p"]["d"] == 1


def test_mock_duel_end_to_end(tmp_path):
    g = get_game("sudoku3d")
    dc = DuelConfig(game="sudoku3d", game_config={"n": 4}, puzzles_per_side=3, make_time_s=30,
                    solve_time_s=30, out_dir=str(tmp_path), leaderboard=str(tmp_path / "lb.json"))
    a = make_player({"type": "mock", "id": "strong", "skill": 1.0, "difficulty": "hard", "seed": 1}, g)
    b = make_player({"type": "mock", "id": "weak", "skill": 0.0, "bad_make": 1.0, "seed": 2}, g)
    res = run_duel(g, a, b, dc)
    assert res["winner"] == "a"
    assert res["scores"]["b"]["n_rejected"] == 3 and res["scores"]["a"]["n_correct"] == 0
    assert res["scores"]["b"]["n_correct"] == 0
    lines = (tmp_path / res["duel_id"] / "replay.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["event"] == "start" and json.loads(lines[-1])["event"] == "end"
    assert json.loads((tmp_path / "lb.json").read_text())["sudoku3d"]["strong"]["w"] == 1


def test_command_adapter(tmp_path):
    bot = tmp_path / "bot.py"
    bot.write_text("import json,sys; r=json.load(sys.stdin); print(json.dumps({'echo': r['phase']}))")
    g = get_game("sudoku3d")
    rep = CommandPlayer(["python", str(bot)]).act(make_request(g, g.parse_config({"n": 4}), 1, 10))
    assert rep.data == {"echo": "make"} and not rep.error


def test_openai_adapter_against_fake_server():
    """Spin a tiny local OpenAI-compatible server that answers with a fixed JSON."""
    import http.server, threading
    from gamekhelega.engine.adapters.openai_compat import OpenAICompatPlayer

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["messages"][0]["role"] == "system"
            out = {"choices": [{"message": {"content": '```json\n{"solution": null}\n```'}}],
                   "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    g = get_game("sudoku3d")
    p = OpenAICompatPlayer(model="fake", base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env=None)
    rep = p.act(solve_request(g, g.parse_config({"n": 4}), "a1", {"layers": []}, 10))
    srv.shutdown()
    assert rep.data == {"solution": None} and rep.usage["prompt_tokens"] == 7


def test_human_adapter_scripted():
    from rich.console import Console
    g = get_game("sudoku3d")
    cfg = g.parse_config({"n": 4})
    p, s = g.generate(cfg, "medium", np.random.default_rng(9))
    cmds = [f"{x+1} {y+1} {z+1} {s[z, y, x]}" for z, y, x in np.argwhere(p == 0)] + ["submit"]
    it = iter(cmds)
    hp = make_player({"type": "human", "id": "me"}, g)
    hp.input, hp.console = (lambda _p: next(it)), Console(file=open("/dev/null", "w"))
    rep = hp.act(solve_request(g, cfg, "x", g.puzzle_to_json(cfg, p), 60))
    assert g.check_solution(cfg, p, g.solution_from_json(cfg, rep.data["solution"])).ok


def test_template_game_plugs_in():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))
    from minimal_game import GAME as latin
    from gamekhelega.engine.adapters.mock import MockPlayer
    import tempfile
    cfg = latin.parse_config({"n": 4})
    p, s = latin.generate(cfg, "medium", np.random.default_rng(0))
    assert latin.count_solutions(cfg, p) == 1
    with tempfile.TemporaryDirectory() as d:
        res = run_duel(latin, MockPlayer(latin, "x"), MockPlayer(latin, "y", skill=0.0),
                       DuelConfig(game="latin", game_config={"n": 4}, puzzles_per_side=2, make_time_s=20,
                                  solve_time_s=20, out_dir=d, leaderboard=None))
    assert res["winner"] == "a"
