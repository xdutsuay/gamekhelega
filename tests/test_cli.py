from typer.testing import CliRunner

from gamekhelega.cli import app

runner = CliRunner()


def test_cli_generate_check_view_solve(tmp_path):
    f = tmp_path / "p.json"
    r = runner.invoke(app, ["generate", "--game", "sudoku3d", "-p", "n=4", "--count", "2", "--seed", "1",
                            "--out", str(f)])
    assert r.exit_code == 0, r.output
    assert runner.invoke(app, ["check", str(f)]).exit_code == 0
    assert runner.invoke(app, ["solve", str(f)]).exit_code == 0
    r = runner.invoke(app, ["view", str(f), "--html", str(tmp_path / "v.html")])
    assert r.exit_code == 0 and (tmp_path / "v.html").exists()
    assert "sudoku3d" in runner.invoke(app, ["games"]).output


def test_cli_duel_and_replay(tmp_path):
    import json
    cfg = json.loads(open("configs/duel_mock_4.json").read())
    cfg["out_dir"] = str(tmp_path); cfg["leaderboard"] = str(tmp_path / "lb.json")
    c = tmp_path / "d.json"; c.write_text(json.dumps(cfg))
    r = runner.invoke(app, ["duel", str(c)])
    assert r.exit_code == 0, r.output
    run = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert runner.invoke(app, ["replay", str(run), "--boards"]).exit_code == 0
    assert "mock-strong" in runner.invoke(app, ["leaderboard", "--path", str(tmp_path / "lb.json")]).output


def test_every_command_accepts_game():
    for cmd in ["generate", "solve", "check", "play", "duel", "view", "replay", "leaderboard"]:
        out = runner.invoke(app, [cmd, "--help"], env={"COLUMNS": "200"}).output
        assert "--game" in out, cmd
