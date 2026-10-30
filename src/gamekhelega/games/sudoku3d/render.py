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


GIVEN_COLOR = "#111827"    # near-black
FILLED_COLOR = "#2563eb"   # blue
EMPTY_COLOR = "#9ca3af"    # grey
EXPLODE_GAP = 1.5          # extra spacing between slices in the exploded view


def _view_cells(n: int, axis: str, mode: str, k: int | None = None):
    """Cells shown in one view as 0-based (z, y, x) index arrays, plus plotted positions.

    mode: "all" (whole cube), "exploded" (whole cube, slices spread along `axis`),
    "slice" (only slice k along `axis`)."""
    zz, yy, xx = (a.ravel() for a in np.indices((n, n, n)))
    coord = {"z": zz, "y": yy, "x": xx}
    sel = np.ones(zz.shape, dtype=bool) if mode != "slice" else (coord[axis] == k)
    pos = {a: coord[a][sel].astype(float) + 1 for a in "zyx"}
    if mode == "exploded":
        pos[axis] = coord[axis][sel] * (1 + EXPLODE_GAP) + 1
    return zz[sel], yy[sel], xx[sel], pos


def _axis_layout(n: int, axis: str, mode: str) -> dict:
    """Scene axes: labelled x/y/z, 1-based ticks, fixed ranges (stable camera between frames)."""
    ticks = list(range(1, n + 1))
    scene = {}
    for a in "xyz":
        vals = ticks
        if mode == "exploded" and a == axis:
            vals = [k * (1 + EXPLODE_GAP) + 1 for k in range(n)]
        scene[f"{a}axis"] = dict(title=a, tickvals=vals, ticktext=[str(t) for t in ticks],
                                 range=[0.5, vals[-1] + 0.5], autorange=False)
    scene["aspectmode"] = "manual"
    scene["aspectratio"] = {a: (1.6 if mode == "exploded" and a == axis else 1.0) for a in "xyz"}
    return scene


def render_3d(cfg: Rules, puzzle, solution=None):
    """Interactive plotly cube.

    - Traces: givens (dark, large), solver-filled cells (blue; only when `solution` is given),
      empty cells (faint small grey markers).
    - Dropdown "slice axis": z-layers / y-slices / x-slices.
    - Slider: ``all``, ``exploded`` (slices spread apart along the axis), then one slice ``1..n``.
    - With a solution, a toggle shows/hides the filled cells.
    - Hover: ``x, y, z = value (given|filled|empty)`` with 1-based coordinates.
    Views switch via plotly "update" calls (no animation frames), so the exported HTML opens
    on the whole cube instead of auto-playing through frames.
    Returns plotly.graph_objects.Figure."""
    import plotly.graph_objects as go

    n = cfg.n
    puzzle = np.asarray(puzzle)
    grid = np.asarray(solution) if solution is not None else puzzle
    givens = puzzle > 0
    filled = (grid > 0) & ~givens
    has_filled = solution is not None
    kinds = [("given", givens), ("filled", filled), ("empty", grid == 0)]

    def view_data(axis: str, mode: str, k: int | None = None) -> list[dict]:
        zi, yi, xi, pos = _view_cells(n, axis, mode, k)
        out = []
        for status, mask in kinds:
            m = mask[zi, yi, xi]
            v = grid[zi[m], yi[m], xi[m]]
            label = [ALPHABET[int(t) - 1] if t else "·" for t in v]
            out.append(dict(
                x=pos["x"][m].tolist(), y=pos["y"][m].tolist(), z=pos["z"][m].tolist(),
                text=label if status != "empty" else None,
                customdata=[[int(a) + 1, int(b) + 1, int(c) + 1, lab]
                            for a, b, c, lab in zip(xi[m], yi[m], zi[m], label)]))
        return out

    hover = "x, y, z = %{customdata[0]}, %{customdata[1]}, %{customdata[2]} = %{customdata[3]} (STATUS)<extra></extra>"
    styles = {
        "given": dict(mode="markers+text", marker=dict(size=7, color=GIVEN_COLOR, opacity=0.95),
                      textfont=dict(size=13, color=GIVEN_COLOR, family="Arial Black, Arial, sans-serif")),
        "filled": dict(mode="markers+text", marker=dict(size=4, color=FILLED_COLOR, opacity=0.6),
                       textfont=dict(size=11, color=FILLED_COLOR)),
        "empty": dict(mode="markers", marker=dict(size=2, color=EMPTY_COLOR, opacity=0.25)),
    }

    def step_names(axis):
        return ["all", "exploded"] + [str(k + 1) for k in range(n)]

    def view_update(axis: str, key: str) -> list:
        """args for a plotly "update" call: new trace data + axis layout as dotted keys
        (dotted keys keep the user's camera; no frames, so write_html's auto_play has nothing to play)."""
        mode = key if key in ("all", "exploded") else "slice"
        data = view_data(axis, mode, int(key) - 1 if mode == "slice" else None)
        lay = {f"scene.{ax}.{k}": v for ax, d in _axis_layout(n, axis, mode).items()
               if isinstance(d, dict) for k, v in d.items()}
        return [{f: [t[f] for t in data] for f in ("x", "y", "z", "text", "customdata")}, lay]

    initial = view_data("z", "all")
    fig = go.Figure(data=[go.Scatter3d(name=status, hovertemplate=hover.replace("STATUS", status),
                                       visible=True if (status != "filled" or has_filled) else "legendonly",
                                       **styles[status], **d)
                          for (status, _), d in zip(kinds, initial)])

    axis_title = {"z": "z-layer", "y": "y-slice", "x": "x-slice"}

    def slider(axis: str) -> dict:
        steps = [dict(method="update", label=key, args=view_update(axis, key)) for key in step_names(axis)]
        return dict(active=0, currentvalue=dict(prefix=f"{axis_title[axis]}: "), steps=steps,
                    pad=dict(t=30), len=0.9, x=0.05)

    sliders = {axis: slider(axis) for axis in ("z", "y", "x")}
    axis_buttons = []
    for axis in ("z", "y", "x"):
        data_upd, lay = view_update(axis, "all")
        axis_buttons.append(dict(label=f"{axis_title[axis]}s", method="update",
                                 args=[data_upd, {**lay, "sliders": [sliders[axis]]}]))
    menus = [dict(type="dropdown", name="slice axis", direction="down", x=0.0, y=1.0,
                  xanchor="left", yanchor="bottom", showactive=True, active=0, buttons=axis_buttons)]
    if has_filled:
        menus.append(dict(type="buttons", name="filled toggle", direction="right", x=0.2, y=1.0,
                          xanchor="left", yanchor="bottom", showactive=True, active=0, buttons=[
                              dict(label="show filled", method="restyle", args=[{"visible": True}, [1]]),
                              dict(label="hide filled", method="restyle", args=[{"visible": False}, [1]]),
                          ]))

    givens_n, filled_n = int(givens.sum()), int(filled.sum())
    fig.update_layout(
        title=dict(text=f"3D Sudoku {n}³ ({cfg.variant}): {givens_n} givens"
                   + (f", {filled_n} filled" if has_filled else ""), x=0.5, xanchor="center", y=0.98),
        scene=_axis_layout(n, "z", "all"),
        sliders=[sliders["z"]], updatemenus=menus,
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=100, b=0))
    return fig
