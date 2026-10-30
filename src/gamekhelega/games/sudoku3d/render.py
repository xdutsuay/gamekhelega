"""Terminal (rich) and 3D (plotly) rendering for sudoku3d."""
from __future__ import annotations

import numpy as np
from rich.columns import Columns
from rich.table import Table
from rich.text import Text

from .rules import Rules, conflicts

ALPHABET = "123456789ABCDEFG"
AXES = {"z": 0, "y": 1, "x": 2}


def _slice(g: np.ndarray, axis: str, k: int) -> np.ndarray:
    """2D view of slice k along axis; rows/cols: z->(y,x), y->(z,x), x->(z,y)."""
    return np.take(g, k, axis=AXES[axis])


def _slice_text(cfg: Rules, grid, givens, bad, axis: str, k: int) -> Text:
    b, n = cfg.b, cfg.n
    G, Gv, B = _slice(grid, axis, k), _slice(givens, axis, k), _slice(bad, axis, k)
    rows_lbl, cols_lbl = {"z": ("y", "x"), "y": ("z", "x"), "x": ("z", "y")}[axis]
    t = Text(f"{axis}={k + 1}  ({rows_lbl}↓ {cols_lbl}→)\n", style="bold")
    for r in range(n):
        if r and r % b == 0:
            t.append("─" * (2 * n + 2 * (b - 1)) + "\n", style="dim")
        for c in range(n):
            if c and c % b == 0:
                t.append("│ ", style="dim")
            v = int(G[r, c])
            ch = ALPHABET[v - 1] if v else "·"
            style = "bold red" if B[r, c] else ("bold white" if Gv[r, c] else ("cyan" if v else "dim"))
            t.append(ch + " ", style=style)
        t.append("\n")
    return t


def render_terminal(cfg: Rules, puzzle, solution=None, state=None):
    """Givens bold white, player entries cyan, conflicts red, empty '·'.
    Shows all slices along the chosen axis side by side (wrapped by terminal width),
    or one slice if state.layer is set."""
    givens = np.asarray(puzzle) > 0
    grid = state.grid if state is not None else (solution if solution is not None else puzzle)
    axis = state.axis if state is not None else "z"
    layer = state.layer if state is not None else None
    bad = conflicts(grid, cfg)
    ks = range(cfg.n) if layer is None else [layer]
    panels = [_slice_text(cfg, grid, givens, bad, axis, k) for k in ks]
    filled = int((np.asarray(grid) > 0).sum())
    title = Text(f"{filled}/{cfg.ncells} filled, {int(givens.sum())} givens, {int(bad.sum())} conflicts",
                 style="italic")
    tbl = Table.grid()
    tbl.add_row(title)
    tbl.add_row(Columns(panels, padding=(0, 3)))
    return tbl


def render_3d(cfg: Rules, puzzle, solution=None):
    """Plotly figure: cube of numbered markers (givens large/opaque, solved cells small),
    a slider isolates one z-layer. Returns plotly.graph_objects.Figure."""
    import plotly.graph_objects as go

    grid = np.asarray(solution if solution is not None else puzzle)
    givens = np.asarray(puzzle) > 0
    z, y, x = np.nonzero(grid > 0)
    vals = grid[z, y, x]
    is_g = givens[z, y, x]

    def trace(mask, name, size, opacity):
        return go.Scatter3d(
            x=x[mask] + 1, y=y[mask] + 1, z=z[mask] + 1, mode="markers+text", name=name,
            text=[ALPHABET[v - 1] for v in vals[mask]], textfont=dict(size=10),
            marker=dict(size=size, color=vals[mask], colorscale="Turbo", cmin=1, cmax=cfg.n,
                        opacity=opacity), hovertemplate="x=%{x} y=%{y} z=%{z} v=%{text}")

    def traces(layer):
        sel = np.ones_like(z, dtype=bool) if layer is None else (z == layer)
        return [trace(sel & is_g, "givens", 7, 0.95), trace(sel & ~is_g, "filled", 4, 0.5)]

    fig = go.Figure(data=traces(None))
    fig.frames = [go.Frame(data=traces(k), name=str(k + 1)) for k in range(cfg.n)]
    steps = [dict(method="animate", label="all", args=[[None], dict(mode="immediate")])]
    steps = [dict(method="animate", label=str(k + 1),
                  args=[[str(k + 1)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
             for k in range(cfg.n)]
    fig.update_layout(
        title=f"3D Sudoku {cfg.n}³ ({cfg.variant})",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        sliders=[dict(active=0, currentvalue=dict(prefix="z-layer "), steps=steps)],
        updatemenus=[dict(type="buttons", showactive=False, buttons=[dict(
            label="all layers", method="update", args=[{"x": [d.x for d in fig.data],
                                                       "y": [d.y for d in fig.data],
                                                       "z": [d.z for d in fig.data],
                                                       "text": [d.text for d in fig.data]}])])])
    return fig
