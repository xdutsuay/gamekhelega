import json
import sys
from pathlib import Path

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


def _verdict_view(g, cfg, v):
    return {
        "accepted": v.accepted, "reason": v.reason, "errors": v.errors,
        "difficulty": v.difficulty,
        "puzzle": None if v.puzzle is None else g.puzzle_to_json(cfg, v.puzzle),
        "solution": None if v.solution is None else g.solution_to_json(cfg, v.solution),
    }


def test_parallel_validation_matches_sequential():
    """A process pool must return the same verdicts, in the same order, as a
    one-by-one loop. Each item still has its own uniqueness time cap."""
    from gamekhelega.engine.validate import check_submission, check_submissions
    g = get_game("sudoku3d")
    cfg = g.parse_config({"n": 4})
    rng = np.random.default_rng(0)
    items = []
    for _ in range(3):
        puz, sol = g.generate(cfg, "medium", rng)
        items.append({"puzzle": g.puzzle_to_json(cfg, puz), "solution": g.solution_to_json(cfg, sol)})
    # A malformed item in the middle must stay in the middle.
    items.insert(1, {"puzzle": "x"})
    empty = np.zeros(cfg.shape, np.int8)
    items.append({"puzzle": g.puzzle_to_json(cfg, empty), "solution": items[0]["solution"]})
    seq = [check_submission(g, cfg, it, uniqueness_time_limit=5) for it in items]
    par = check_submissions(g, cfg, items, uniqueness_time_limit=5, workers=2)
    assert [_verdict_view(g, cfg, v) for v in par] == [_verdict_view(g, cfg, v) for v in seq]
    assert [v.reason for v in par][1] == "malformed"
    assert "not_unique" in [v.reason for v in par]


def test_prompts_and_json_extraction():
    g = get_game("sudoku3d")
    cfg = g.parse_config({"n": 9})
    mk = user_prompt(make_request(g, cfg, 3, 600))
    assert "TASK (MAKE)" in mk and "exactly one solution" in mk and '"layers"' in mk
    sv = user_prompt(solve_request(g, cfg, "a1", {"layers": []}, 60))
    assert "TASK (SOLVE)" in sv and "a1" in sv
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"solution": null} hope it helps') == {"solution": None}


def test_extract_json_skips_decoys_and_prefers_schema():
    """Untrusted model text: decoy braces, a scratch object, then the real answer."""
    decoy = 'Here is {example} and then {"solution": [1, 2]}'
    assert extract_json(decoy) == {"solution": [1, 2]}
    # Two valid objects: the last one that has the expected keys wins, even if
    # a earlier scratch object also parsed.
    two = 'Scratch: {"note": "thinking"} Final: {"solution": null}'
    assert extract_json(two, required=["solution"]) == {"solution": None}
    assert extract_json(two, schema={"required": ["solution"]}) == {"solution": None}
    # A fenced ```json block is preferred over an earlier prose object.
    fenced = 'Prose {"solution": "wrong"}\n```json\n{"solution": "right"}\n```'
    assert extract_json(fenced, required=["solution"]) == {"solution": "right"}
    # A plain fence still beats following prose if it matches.
    plain = '```\n{"solution": 1}\n``` and later {"solution": 2}'
    assert extract_json(plain, required=["solution"]) == {"solution": 1}
    # Nested braces in a string stay inside the parent object.
    nested = '{"solution": "cell {1,2}"} trailing {"note": "nope"}'
    assert extract_json(nested, required=["solution"]) == {"solution": "cell {1,2}"}
    # No object at all is a clean error, not a JSONDecodeError on a decoy.
    try:
        extract_json("sorry, {example} is not json")
    except ValueError as e:
        assert "no JSON object" in str(e)
    else:
        raise AssertionError("expected ValueError")


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


def test_elo_concurrent_writes_lose_no_updates(tmp_path):
    """Many writers at once must all land: wins + losses add up, file stays valid JSON."""
    import threading
    path = tmp_path / "lb.json"
    n = 40
    errors = []

    def one():
        try:
            elo.record(path, "g", "p", "q", "a")
        except Exception as e:  # noqa: BLE001 — collect and fail after join
            errors.append(e)

    threads = [threading.Thread(target=one) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    board = json.loads(path.read_text())["g"][""]
    assert board["p"]["w"] == n and board["q"]["l"] == n
    assert board["p"]["l"] == 0 and board["q"]["w"] == 0
    assert not list(tmp_path.glob("*.tmp.*"))
    assert not list(tmp_path.glob("*.lock"))


def test_elo_rebuild_matches_incremental(tmp_path):
    """A leaderboard rebuilt from result.json files must match one built as the duels ran.

    Files are applied in duel_id order, not directory or mtime order, so a later
    write with an earlier id still lands in the right place.
    """
    runs = tmp_path / "runs"
    # Deliberately not in chronological directory-creation or duel_id order.
    events = [
        ("20260101-000003-cccccc", "g", "r", "q", "b"),
        ("20260101-000001-aaaaaa", "g", "p", "q", "a"),
        ("20260101-000002-bbbbbb", "g", "p", "r", "draw"),
    ]
    for did, game, a, b, w in events:
        d = runs / did
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "duel_id": did, "game": game, "players": {"a": a, "b": b}, "winner": w,
        }))
    lb = tmp_path / "lb.json"
    for did, game, a, b, w in sorted(events, key=lambda e: e[0]):
        elo.record(lb, game, a, b, w)
    rebuilt = tmp_path / "rebuilt.json"
    elo.rebuild(runs, rebuilt)
    assert json.loads(lb.read_text()) == json.loads(rebuilt.read_text())


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
    assert json.loads((tmp_path / "lb.json").read_text())["sudoku3d"]["n=4,variant=layer"]["strong"]["w"] == 1


def _stable_duel(res: dict) -> dict:
    """Drop id and timings so two seeded runs can be compared."""
    puzzles = {
        k: [{kk: vv for kk, vv in rec.items()} for rec in recs]
        for k, recs in res["puzzles"].items()
    }
    solves = {
        k: [{kk: vv for kk, vv in r.items() if kk != "time_s"} for r in recs]
        for k, recs in res["solves"].items()
    }
    scores = {
        k: {kk: vv for kk, vv in s.items() if kk not in ("solve_time_s", "solve_pts", "total")}
        for k, s in res["scores"].items()
    }
    return {"winner": res["winner"], "puzzles": puzzles, "solves": solves, "scores": scores}


def test_duel_seed_is_reproducible(tmp_path):
    """The same DuelConfig.seed must produce the same puzzles, verdicts and scores."""
    g = get_game("sudoku3d")

    def once(folder):
        dc = DuelConfig(game="sudoku3d", game_config={"n": 4}, puzzles_per_side=2,
                        make_time_s=20, solve_time_s=20, seed=123,
                        out_dir=str(folder), leaderboard=None)
        a = make_player({"type": "mock", "id": "a", "skill": 0.5, "difficulty": "medium", "seed": 1}, g)
        b = make_player({"type": "mock", "id": "b", "skill": 0.5, "bad_make": 0.3, "seed": 2}, g)
        return run_duel(g, a, b, dc)

    r1, r2 = once(tmp_path / "r1"), once(tmp_path / "r2")
    assert r1["duel_id"] != r2["duel_id"]
    assert _stable_duel(r1) == _stable_duel(r2)
    # A different seed should change the puzzles (otherwise the seed still isn't doing anything).
    dc = DuelConfig(game="sudoku3d", game_config={"n": 4}, puzzles_per_side=2,
                    make_time_s=20, solve_time_s=20, seed=999,
                    out_dir=str(tmp_path / "r3"), leaderboard=None)
    a = make_player({"type": "mock", "id": "a", "skill": 0.5, "difficulty": "medium", "seed": 1}, g)
    b = make_player({"type": "mock", "id": "b", "skill": 0.5, "bad_make": 0.3, "seed": 2}, g)
    r3 = run_duel(g, a, b, dc)
    assert _stable_duel(r1)["puzzles"] != _stable_duel(r3)["puzzles"]


def test_command_adapter(tmp_path):
    bot = tmp_path / "bot.py"
    bot.write_text("import json,sys; r=json.load(sys.stdin); print(json.dumps({'echo': r['phase']}))")
    g = get_game("sudoku3d")
    rep = CommandPlayer([sys.executable, str(bot)]).act(make_request(g, g.parse_config({"n": 4}), 1, 10))
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


def test_openai_default_max_tokens_is_a_guardrail():
    from gamekhelega.engine.adapters.openai_compat import DEFAULT_MAX_TOKENS, OpenAICompatPlayer
    ok = {"choices": [{"message": {"content": '{"solution": null}'}}], "usage": {}}
    g = get_game("sudoku3d")
    req = solve_request(g, g.parse_config({"n": 4}), "a1", {"layers": []}, 10)
    srv, hits = _openai_scripted_server([(200, {}, ok)])
    OpenAICompatPlayer(model="fake", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                       api_key_env=None).act(req)
    srv.shutdown()
    assert hits["bodies"][0]["max_tokens"] == DEFAULT_MAX_TOKENS == 8192
    srv, hits = _openai_scripted_server([(200, {}, ok)])
    OpenAICompatPlayer(model="fake", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                       api_key_env=None, max_tokens=None).act(req)
    srv.shutdown()
    assert "max_tokens" not in hits["bodies"][0]


def _openai_scripted_server(script):
    """HTTP server that walks a list of (status, headers, body) responses."""
    import http.server, threading
    hits = {"n": 0, "bodies": []}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            hits["bodies"].append(json.loads(raw))
            i = min(hits["n"], len(script) - 1)
            hits["n"] += 1
            status, headers, body = script[i]
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, hits


def test_openai_retries_backoff_and_keeps_usage():
    """401 is not retried; 429 backs off (Retry-After) and usage from every attempt is kept.
    A 400 json_mode fallback must not permanently flip the player."""
    import time
    from gamekhelega.engine.adapters.openai_compat import OpenAICompatPlayer

    ok = {"choices": [{"message": {"content": '{"solution": null}'}}],
          "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    g = get_game("sudoku3d")
    req = solve_request(g, g.parse_config({"n": 4}), "a1", {"layers": []}, 10)

    # 401 is a dead end: retries=3 must still hit the server once.
    srv, hits = _openai_scripted_server([
        (401, {}, {"error": {"message": "nope"}, "usage": {"prompt_tokens": 2}}),
    ])
    p = OpenAICompatPlayer(model="fake", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                           api_key_env=None, retries=3)
    rep = p.act(req)
    srv.shutdown()
    assert hits["n"] == 1 and "401" in rep.error
    assert rep.usage["prompt_tokens"] == 2

    # 429 then 200: honour Retry-After, add usage from both attempts.
    srv, hits = _openai_scripted_server([
        (429, {"Retry-After": "0.3"}, {"error": "slow down", "usage": {"prompt_tokens": 5}}),
        (200, {}, ok),
    ])
    p = OpenAICompatPlayer(model="fake", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                           api_key_env=None, retries=2)
    t0 = time.monotonic()
    rep = p.act(req)
    dt = time.monotonic() - t0
    srv.shutdown()
    assert rep.data == {"solution": None}
    assert hits["n"] == 2
    assert dt >= 0.3
    assert rep.usage["prompt_tokens"] == 12 and rep.usage["completion_tokens"] == 3

    # 400 on json_mode: retry immediately without response_format, do not flip the player.
    srv, hits = _openai_scripted_server([
        (400, {}, {"error": "no json_object"}),
        (200, {}, ok),
    ])
    p = OpenAICompatPlayer(model="fake", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                           api_key_env=None, retries=1, json_mode=True)
    rep = p.act(req)
    srv.shutdown()
    assert p.json_mode is True
    assert "response_format" in hits["bodies"][0] and "response_format" not in hits["bodies"][1]
    assert rep.data == {"solution": None}


def test_openai_timeout_cancels_http_request():
    """A slow server must be disconnected near the deadline, not left running.

    The server dribbles the body one byte every 0.25s so a per-operation
    socket timeout would never fire; only closing the connection (cancel)
    cuts the request off.
    """
    import http.server, threading, time
    from gamekhelega.engine.adapters.openai_compat import OpenAICompatPlayer
    from gamekhelega.engine.duel import _timed

    state = {"finished": False}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            out = json.dumps({"choices": [{"message": {"content": '{"solution": null}'}}],
                              "usage": {"prompt_tokens": 99, "completion_tokens": 1}}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                for i in range(0, len(out)):
                    self.wfile.write(out[i:i + 1])
                    self.wfile.flush()
                    time.sleep(0.25)
                state["finished"] = True
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    g = get_game("sudoku3d")
    player = OpenAICompatPlayer(model="fake", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                api_key_env=None, retries=0)
    # 1.5s is above the adapter's "remaining <= 1s, don't start" guard.
    req = solve_request(g, g.parse_config({"n": 4}), "a1", {"layers": []}, 1.5)
    t0 = time.monotonic()
    rep, dt = _timed(player, req)
    elapsed = time.monotonic() - t0
    time.sleep(0.8)
    srv.shutdown()
    assert rep.error == "timeout", rep
    assert "cancelled" in (rep.usage or {})
    # Deadline 1.5s + 1s grace + slack; a dribble of the full body would take many seconds.
    assert elapsed < 3.0, elapsed
    assert dt < 3.0, dt
    assert not state["finished"], "server completed the response after the client should have hung up"


def test_command_adapter_kills_subprocess(tmp_path):
    """subprocess.run(..., timeout=) must actually kill the child, not abandon it."""
    import os, time
    marker = tmp_path / "pid.txt"
    bot = tmp_path / "bot.py"
    bot.write_text(
        "import os, sys, time, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    g = get_game("sudoku3d")
    req = make_request(g, g.parse_config({"n": 4}), 1, 0.4)
    t0 = time.monotonic()
    rep = CommandPlayer([sys.executable, str(bot)]).act(req)
    elapsed = time.monotonic() - t0
    assert rep.error == "timeout"
    assert elapsed < 2.0, elapsed
    # The child wrote its pid, then slept; it must be gone now.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    if marker.exists():
        pid = int(marker.read_text())
        gone = False
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except OSError:
                gone = True
                break
            time.sleep(0.05)
        assert gone, f"child pid {pid} still running"


def test_command_adapter_honours_cancel_and_kills_the_group(tmp_path):
    """req.cancel must stop the player near now, and kill grandchildren too."""
    import os, threading, time
    marker = tmp_path / "pids.txt"
    bot = tmp_path / "bot.py"
    bot.write_text(
        "import os, sys, time, pathlib, subprocess\n"
        f"path = pathlib.Path({str(marker)!r})\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "path.write_text(f'{os.getpid()} {child.pid}')\n"
        "time.sleep(30)\n"
    )
    g = get_game("sudoku3d")
    req = make_request(g, g.parse_config({"n": 4}), 1, 30)  # long budget; cancel should win
    req.cancel = threading.Event()
    box: dict = {}

    def run():
        box["rep"] = CommandPlayer([sys.executable, str(bot)]).act(req)

    th = threading.Thread(target=run)
    th.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)
    req.cancel.set()
    t0 = time.monotonic()
    th.join(3)
    elapsed = time.monotonic() - t0
    assert not th.is_alive()
    assert box["rep"].error == "timeout"
    assert elapsed < 1.5, elapsed
    assert marker.exists(), "child never wrote its pids"
    pids = [int(x) for x in marker.read_text().split()]
    for pid in pids:
        gone = False
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except OSError:
                gone = True
                break
            time.sleep(0.05)
        assert gone, f"pid {pid} still running after cancel"


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


def test_wilson_interval_is_honest_about_one_game():
    from gamekhelega.engine.match import wilson_interval
    lo, hi = wilson_interval(1, 1)
    assert 0 <= lo < 0.5 and hi == 1.0
    lo0, hi0 = wilson_interval(0, 1)
    assert lo0 == 0.0 and 0.5 < hi0 <= 1.0
    lo10, hi10 = wilson_interval(10, 10)
    assert lo10 > lo  # more wins shrink the uncertainty on the low side
    assert hi10 == 1.0


def test_match_swaps_seats_and_reports_ci(tmp_path):
    from gamekhelega.engine.match import run_match
    g = get_game("sudoku3d")
    dc = DuelConfig(game="sudoku3d", game_config={"n": 4}, puzzles_per_side=1,
                    make_time_s=15, solve_time_s=15, seed=7,
                    out_dir=str(tmp_path), leaderboard=None)
    a = make_player({"type": "mock", "id": "strong", "skill": 1.0, "difficulty": "medium"}, g)
    b = make_player({"type": "mock", "id": "weak", "skill": 0.0, "bad_make": 1.0}, g)
    m = run_match(g, a, b, dc, games=4)
    assert m["a_wins"] + m["b_wins"] + m["draws"] == 4
    assert m["a_wins"] == 4  # weak always makes invalid puzzles and can't solve
    assert m["duels"][0]["seats"] == {"a": "strong", "b": "weak"}
    assert m["duels"][1]["seats"] == {"a": "weak", "b": "strong"}
    assert m["duels"][0]["seed"] == 7 and m["duels"][1]["seed"] == 8
    lo, hi = m["ci95"]
    assert 0 <= lo <= m["a_rate"] <= hi <= 1
    assert Path(m["out"]).exists()
    # Same seeds and players again: same mapped winners (reproducible match).
    m2 = run_match(g, a, b, DuelConfig(game="sudoku3d", game_config={"n": 4}, puzzles_per_side=1,
                                       make_time_s=15, solve_time_s=15, seed=7,
                                       out_dir=str(tmp_path / "r2"), leaderboard=None), games=4)
    assert [d["score_a"] for d in m["duels"]] == [d["score_a"] for d in m2["duels"]]


def test_bt_recovers_ranking_and_is_order_independent(tmp_path):
    """A known transitive set of results should rank A > B > C with a tight CI
    at the top, and the fit must not depend on the order of events (unlike Elo)."""
    from gamekhelega.engine import bt, elo
    # 20 games on each edge of A>B>C.
    evs = ([("A", "B", 1.0)] * 20 + [("A", "C", 1.0)] * 20 + [("B", "C", 1.0)] * 20)
    fit = bt.bootstrap(evs, n_boot=200, seed=0)
    assert fit["A"]["rating"] > fit["B"]["rating"] > fit["C"]["rating"]
    assert fit["A"]["lo"] <= fit["A"]["rating"] <= fit["A"]["hi"]
    assert fit["A"]["lo"] > fit["C"]["hi"]  # clearly separated
    assert fit["A"]["w"] == 40 and fit["C"]["l"] == 40

    # Same results in two orders: BT is identical, online Elo is not.
    upset_then_wins = [("weak", "strong", 1.0)] + [("strong", "weak", 1.0)] * 6
    wins_then_upset = list(reversed(upset_then_wins))
    bt1 = bt.fit(upset_then_wins)[2]
    bt2 = bt.fit(wins_then_upset)[2]
    assert list(bt.fit(upset_then_wins)[0]) == list(bt.fit(wins_then_upset)[0])
    assert list(bt1) == list(bt2)
    p1, p2 = tmp_path / "e1.json", tmp_path / "e2.json"
    for a, b, sa in upset_then_wins:
        elo.record(p1, "g", a, b, {1.0: "a", 0.0: "b", 0.5: "draw"}[sa])
    for a, b, sa in wins_then_upset:
        elo.record(p2, "g", a, b, {1.0: "a", 0.0: "b", 0.5: "draw"}[sa])
    e1, e2 = json.loads(p1.read_text())["g"][""], json.loads(p2.read_text())["g"][""]
    assert e1["strong"]["elo"] != e2["strong"]["elo"]


def test_bt_reads_result_files_and_cli(tmp_path):
    from typer.testing import CliRunner
    from gamekhelega.cli import app
    from gamekhelega.engine import bt
    runs = tmp_path / "runs"
    evs = [
        ("20260101-000001-aaaaaa", "g", "A", "B", "a"),
        ("20260101-000002-bbbbbb", "g", "A", "C", "a"),
        ("20260101-000003-cccccc", "g", "B", "C", "a"),
    ]
    for i, (did, game, a, b, w) in enumerate(evs * 5):
        d = runs / f"{did}-{i}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "duel_id": f"{did}-{i}", "game": game,
            "players": {"a": a, "b": b}, "winner": w,
        }))
    data = bt.fit_runs(runs, n_boot=50, seed=1)
    assert data["g"][""]["A"]["rating"] > data["g"][""]["B"]["rating"] > data["g"][""]["C"]["rating"]
    out = tmp_path / "bt.json"
    r = CliRunner().invoke(app, ["elo", "bt", "--runs", str(runs), "--boots", "30",
                                 "--out", str(out), "--game", "g"])
    assert r.exit_code == 0, r.output
    assert "Bradley-Terry" in r.output and "A" in r.output
    assert json.loads(out.read_text())["g"][""]["A"]["w"] == 10


def test_bt_ratings_are_per_game_and_size(tmp_path):
    """A wins every n=4 duel and loses every n=9 duel. Pooled by game only,
    the two would cancel; bucketed by size, A is ahead at n=4 and behind at n=9."""
    from gamekhelega.engine import bt
    from gamekhelega.engine.elo import size_key
    runs = tmp_path / "runs"
    s4, s9 = size_key({"n": 4}), size_key({"n": 9})
    for i in range(8):
        for n, winner in ((4, "a"), (9, "b")):
            d = runs / f"{n}-{i}"
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps({
                "duel_id": f"d-{n}-{i}", "game": "sudoku3d",
                "game_config": {"n": n},
                "players": {"a": "A", "b": "B"}, "winner": winner,
            }))
    data = bt.fit_runs(runs, n_boot=40, seed=0)
    assert data["sudoku3d"][s4]["A"]["rating"] > data["sudoku3d"][s4]["B"]["rating"]
    assert data["sudoku3d"][s9]["A"]["rating"] < data["sudoku3d"][s9]["B"]["rating"]
    assert data["sudoku3d"][s4]["A"]["w"] == 8 and data["sudoku3d"][s9]["A"]["l"] == 8


def test_elo_keys_by_size_and_reads_legacy(tmp_path):
    """A 4³ rating is not a 9³ rating. Legacy flat files still load, and a
    sized write migrates them without dropping the old numbers."""
    p = tmp_path / "lb.json"
    elo.record(p, "sudoku3d", "x", "y", "a", size="n=4")
    elo.record(p, "sudoku3d", "x", "y", "a", size="n=9")
    data = json.loads(p.read_text())
    assert data["sudoku3d"]["n=4"]["x"]["w"] == 1
    assert data["sudoku3d"]["n=9"]["x"]["w"] == 1
    elo.record(p, "sudoku3d", "x", "y", "a", size="n=4")
    data = json.loads(p.read_text())
    assert data["sudoku3d"]["n=4"]["x"]["w"] == 2
    assert data["sudoku3d"]["n=9"]["x"]["w"] == 1  # untouched
    # Legacy {game: {player: record}} is readable; first sized write wraps it.
    leg = tmp_path / "legacy.json"
    leg.write_text(json.dumps({"sudoku3d": {"old": {"elo": 1600.0, "w": 3, "l": 0, "d": 0}}}))
    elo.record(leg, "sudoku3d", "old", "new", "a", size="n=4")
    migrated = json.loads(leg.read_text())
    assert migrated["sudoku3d"][""]["old"]["w"] == 3 and migrated["sudoku3d"][""]["old"]["elo"] == 1600.0
    assert migrated["sudoku3d"]["n=4"]["old"]["w"] == 1  # new bucket starts at 1500
    from gamekhelega.engine.results import leaderboard_table
    from io import StringIO
    from rich.console import Console
    buf = StringIO()
    Console(file=buf, width=120, color_system=None).print(leaderboard_table(str(leg)))
    text = buf.getvalue()
    assert "old" in text and "n=4" in text and "1600.0" in text
