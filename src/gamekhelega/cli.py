"""CLI: every command takes --game; game config via --config JSON and/or --param k=v."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console

from .engine.io import dump_pack, load_pack, parse_params
from .engine.registry import all_games, get_game

app = typer.Typer(add_completion=False, help="Puzzle duel arena: AI vs AI (or human) puzzle duels.")
con = Console()

GameOpt = typer.Option(..., "--game", "-g", envvar="GAMEKHELEGA_GAME", help="game plugin name (see `games`)")
CfgOpt = typer.Option(None, "--config", "-c", help="game config as JSON")
ParamOpt = typer.Option(None, "--param", "-p", help="game config key=value (repeatable)")


def _game_cfg(game: str, config: Optional[str], param: Optional[list[str]]):
    g = get_game(game)
    return g, g.parse_config(parse_params(param, config))


def _pack_cfg(pack: dict, game: Optional[str]):
    g = get_game(game or pack["game"])
    return g, g.parse_config(pack["config"])


@app.command()
def games():
    """List installed game plugins."""
    for name, g in sorted(all_games().items()):
        con.print(f"[bold]{name}[/bold]  {g.title}: {g.description}")
        con.print(f"   config schema: {json.dumps(g.config_schema())}")


@app.command()
def generate(game: str = GameOpt, config: Optional[str] = CfgOpt, param: Optional[list[str]] = ParamOpt,
             difficulty: str = "medium", count: int = 1, seed: Optional[int] = None,
             time_budget: float = 30.0, out: Path = Path("puzzles.json"), show: bool = False):
    """Generate puzzles (with solutions) into a pack file."""
    g, cfg = _game_cfg(game, config, param)
    rng = np.random.default_rng(seed)
    items = []
    for i in range(count):
        t = time.monotonic()
        p, s = g.generate(cfg, difficulty, rng, time_budget=time_budget)
        meta = {"difficulty": g.rate(cfg, p), "gen_time_s": round(time.monotonic() - t, 3)}
        items.append({"id": f"p{i + 1}", "puzzle": g.puzzle_to_json(cfg, p),
                      "solution": g.solution_to_json(cfg, s), "meta": meta})
        con.print(f"p{i + 1}: {meta}")
        if show:
            con.print(g.render_terminal(cfg, p))
    dump_pack(out, g.name, g.config_to_json(cfg), items)
    con.print(f"wrote {out}")


@app.command()
def solve(pack_file: Path, game: Optional[str] = typer.Option(None, "--game", "-g"),
          time_limit: float = 60.0, show: bool = True):
    """Solve every puzzle in a pack with the reference solver."""
    pack = load_pack(pack_file)
    g, cfg = _pack_cfg(pack, game)
    for it in pack["items"]:
        p = g.puzzle_from_json(cfg, it["puzzle"])
        t = time.monotonic()
        s = g.solve(cfg, p, time_limit=time_limit)
        con.print(f"{it.get('id')}: {'solved' if s is not None else 'no solution'} in {time.monotonic() - t:.3f}s")
        if show and s is not None:
            con.print(g.render_terminal(cfg, p, s))


@app.command()
def check(pack_file: Path, game: Optional[str] = typer.Option(None, "--game", "-g"),
          time_limit: float = 20.0):
    """Referee check of every item: well-formed, solution valid, exactly one solution."""
    from .engine.validate import check_submission

    pack = load_pack(pack_file)
    g, cfg = _pack_cfg(pack, game)
    bad = 0
    for it in pack["items"]:
        v = check_submission(g, cfg, it, time_limit)
        bad += not v.accepted
        con.print(f"{it.get('id')}: " + ("[green]OK[/green] difficulty=" + str(v.difficulty) if v.accepted
                                         else f"[red]{v.reason}[/red] {v.errors}"))
    raise typer.Exit(1 if bad else 0)


@app.command()
def play(game: Optional[str] = typer.Option(None, "--game", "-g", envvar="GAMEKHELEGA_GAME",
                                           help="game plugin (required unless --puzzle)"), config: Optional[str] = CfgOpt, param: Optional[list[str]] = ParamOpt,
         pack_file: Optional[Path] = typer.Option(None, "--puzzle"), index: int = 0,
         difficulty: str = "easy", seed: Optional[int] = None, time_limit: Optional[float] = None):
    """Human mode: play a generated puzzle (or one from a pack) in the terminal."""
    from .engine.adapters.human import play_loop

    if pack_file:
        pack = load_pack(pack_file)
        g, cfg = _pack_cfg(pack, game)
        p = g.puzzle_from_json(cfg, pack["items"][index]["puzzle"])
    else:
        if not game:
            raise typer.BadParameter("--game is required unless --puzzle is given")
        g, cfg = _game_cfg(game, config, param)
        p, _ = g.generate(cfg, difficulty, np.random.default_rng(seed), time_budget=30)
    t = time.monotonic()
    sol = play_loop(g, cfg, p, input, con, time_limit)
    if sol is None:
        con.print("gave up")
        raise typer.Exit(1)
    r = g.check_solution(cfg, p, sol)
    con.print(f"{'[green]SOLVED[/green]' if r.ok else '[red]not solved[/red]'} in "
              f"{time.monotonic() - t:.0f}s {r.errors}")


@app.command()
def duel(duel_config: Path = typer.Argument(..., help="duel JSON (see configs/)"),
         a: Optional[str] = typer.Option(None, help="override player A spec (JSON)"),
         b: Optional[str] = typer.Option(None, help="override player B spec (JSON)"),
         game: Optional[str] = typer.Option(None, "--game", "-g", help="override game")):
    """Run a duel: both make puzzles, then solve each other's; writes runs/<id>/."""
    from .engine.adapters.base import make_player
    from .engine.duel import DuelConfig, run_duel
    from .engine.results import result_table

    d = json.loads(duel_config.read_text())
    if game:
        d["game"] = game
    dc = DuelConfig.from_dict(d)
    g = get_game(dc.game)
    pa = make_player(json.loads(a) if a else d["player_a"], g)
    pb = make_player(json.loads(b) if b else d["player_b"], g)
    res = run_duel(g, pa, pb, dc)
    con.print(result_table(res))
    con.print(f"results: {res['out_dir']}/result.json  replay: {res['out_dir']}/replay.jsonl")


@app.command()
def view(pack_file: Path, index: int = 0, html: Optional[Path] = None, solution: bool = False,
         game: Optional[str] = typer.Option(None, "--game", "-g", help="override pack's game")):
    """Show a puzzle in the terminal; --html writes an interactive 3D view."""
    from .engine.viewer import export_html, show

    pack = load_pack(pack_file)
    g, cfg = _pack_cfg(pack, game)
    it = pack["items"][index]
    p = g.puzzle_from_json(cfg, it["puzzle"])
    s = g.solution_from_json(cfg, it["solution"]) if solution and it.get("solution") else None
    show(g, cfg, p, s, con)
    if html:
        con.print(f"wrote {export_html(g, cfg, p, s, str(html))}")


@app.command()
def replay(run_dir: Path, boards: bool = False,
           game: Optional[str] = typer.Option(None, "--game", "-g", help="assert the run's game")):
    """Walk through a finished duel's replay log."""
    import json as _json
    from .engine.viewer import replay as _replay
    if game:
        first = _json.loads(open(Path(run_dir) / "replay.jsonl").readline())
        if first.get("game") != game:
            raise typer.BadParameter(f"run is for game {first.get('game')!r}, not {game!r}")
    _replay(str(run_dir), con, boards)


@app.command()
def leaderboard(path: Path = Path("runs/leaderboard.json"),
                game: Optional[str] = typer.Option(None, "--game", "-g", help="only this game")):
    """Show Elo ratings."""
    from .engine.results import leaderboard_table
    con.print(leaderboard_table(str(path), game))


if __name__ == "__main__":
    app()
