"""Terminal (rich) and 3D (plotly) rendering for zip3d."""
from __future__ import annotations

from rich.columns import Columns
from rich.table import Table
from rich.text import Text

from .model import ZipPuzzle, cell, edge, xyz

EXPLODE_GAP = 1.5


def _steps(path) -> dict[int, int]:
    return {c: i for i, c in enumerate(path or [])}


def _layer_text(p: ZipPuzzle, path, z: int, head: int | None) -> Text:
    n = p.n
    step = _steps(path)
    on_path = {edge(a, b) for a, b in zip(path or [], (path or [])[1:])}
    w = max(2, len(str(n ** 3)))
    t = Text(f"z={z + 1}  (y↓ x→)\n", style="bold")
    for y in range(n):
        for x in range(n):
            c = cell(n, x, y, z)
            if c in p.waypoints:
                s, style = f"{p.waypoints[c]:>{w}}", "bold black on yellow"
            elif c in step:
                s, style = f"{step[c] + 1:>{w}}", "cyan"
            else:
                s, style = "·".rjust(w), "dim"
            if c == head:
                style = "bold white on blue"
            t.append(s, style=style)
            # z-links: ↑ next/prev step on z+1, ↓ on z-1, ⇕ both; ▲/▼ walls on z faces
            up = z + 1 < n and edge(c, cell(n, x, y, z + 1)) in on_path
            dn = z > 0 and edge(c, cell(n, x, y, z - 1)) in on_path
            mark = "⇕" if up and dn else ("↑" if up else ("↓" if dn else " "))
            if mark == " ":
                if z + 1 < n and edge(c, cell(n, x, y, z + 1)) in p.walls:
                    mark = "▲"
                elif z > 0 and edge(c, cell(n, x, y, z - 1)) in p.walls:
                    mark = "▼"
            t.append(mark, style="magenta" if mark in "↑↓⇕" else "red")
            if x + 1 < n:
                e = edge(c, cell(n, x + 1, y, z))
                if e in p.walls:
                    t.append("┃", style="bold red")
                elif e in on_path:
                    t.append("─", style="cyan")
                else:
                    t.append(" ")
        t.append("\n")
        if y + 1 < n:
            for x in range(n):
                c = cell(n, x, y, z)
                e = edge(c, cell(n, x, y + 1, z))
                ch, style = ("━", "bold red") if e in p.walls else (("│", "cyan") if e in on_path else (" ", ""))
                t.append(" " * (w - 1) + ch + "  ", style=style)
            t.append("\n")
    return t


def render_terminal(p: ZipPuzzle, path=None, layer: int | None = None):
    """One panel per z-layer: waypoints black-on-yellow, path step numbers cyan with
    ─/│ links inside the layer and ↑/↓ where the path changes layer, walls red (┃ ━ ▲ ▼)."""
    head = path[-1] if path else None
    zs = range(p.n) if layer is None else [layer]
    panels = [_layer_text(p, path, z, head) for z in zs]
    N = p.n ** 3
    title = Text(f"{p.n}³ zip: {p.K} waypoints, {len(p.walls)} walls, path {len(path or [])}/{N} cells",
                 style="italic")
    tbl = Table.grid()
    tbl.add_row(title)
    tbl.add_row(Columns(panels, padding=(0, 3)))
    return tbl


def render_3d(p: ZipPuzzle, path=None):
    """Interactive plotly cube.

    Traces: empty cells (faint), waypoints (large, numbered), solution path (3D line
    coloured by step, start/end marked; toggle show/hide), walls (red squares on faces).
    Slider: ``all``, ``exploded`` (z-layers spread apart), then one z-layer ``1..n``
    (only path segments inside that layer are drawn). Hover shows 1-based x, y, z and step."""
    import plotly.graph_objects as go

    n, N = p.n, p.n ** 3
    step = _steps(path)

    def zpos(z, mode):
        return z * (1 + EXPLODE_GAP) + 1 if mode == "exploded" else z + 1

    def keep(z, mode, k):
        return mode != "slice" or z == k

    def view(mode: str, k: int | None = None):
        cells = {"x": [], "y": [], "z": [], "cd": []}
        wps = {"x": [], "y": [], "z": [], "text": [], "cd": []}
        for c in range(N):
            x, y, z = xyz(n, c)
            if not keep(z, mode, k):
                continue
            tgt = wps if c in p.waypoints else cells
            tgt["x"].append(x + 1); tgt["y"].append(y + 1); tgt["z"].append(zpos(z, mode))
            st = step.get(c)
            tgt["cd"].append([x + 1, y + 1, z + 1, "-" if st is None else st + 1])
            if c in p.waypoints:
                wps["text"].append(str(p.waypoints[c]))
        line = {"x": [], "y": [], "z": [], "color": [], "cd": []}
        if path:
            prev_in = False
            for i, c in enumerate(path):
                x, y, z = xyz(n, c)
                inside = keep(z, mode, k)
                if inside:
                    if i and not prev_in:
                        line["x"].append(None); line["y"].append(None); line["z"].append(None)
                        line["color"].append(i); line["cd"].append([None] * 4)
                    line["x"].append(x + 1); line["y"].append(y + 1); line["z"].append(zpos(z, mode))
                    line["color"].append(i + 1); line["cd"].append([x + 1, y + 1, z + 1, i + 1])
                prev_in = inside
        walls = {"x": [], "y": [], "z": []}
        for a, b in p.walls:
            ax, ay, az = xyz(n, a)
            bx, by, bz = xyz(n, b)
            if not (keep(az, mode, k) or keep(bz, mode, k)):
                continue
            cx, cy = (ax + bx) / 2 + 1, (ay + by) / 2 + 1
            za, zb = zpos(az, mode), zpos(bz, mode)
            cz = (za + zb) / 2
            h = 0.4
            if bx != ax:
                sq = [(cx, cy - h, cz - h), (cx, cy + h, cz - h), (cx, cy + h, cz + h), (cx, cy - h, cz + h)]
            elif by != ay:
                sq = [(cx - h, cy, cz - h), (cx + h, cy, cz - h), (cx + h, cy, cz + h), (cx - h, cy, cz + h)]
            else:
                sq = [(cx - h, cy - h, cz), (cx + h, cy - h, cz), (cx + h, cy + h, cz), (cx - h, cy + h, cz)]
            for q in sq + [sq[0]]:
                walls["x"].append(q[0]); walls["y"].append(q[1]); walls["z"].append(q[2])
            walls["x"].append(None); walls["y"].append(None); walls["z"].append(None)
        return cells, wps, line, walls

    def scene(mode):
        ticks = list(range(1, n + 1))
        zt = [zpos(z, mode) for z in range(n)]
        return dict(
            xaxis=dict(title="x", tickvals=ticks, range=[0.5, n + 0.5], autorange=False),
            yaxis=dict(title="y", tickvals=ticks, range=[0.5, n + 0.5], autorange=False),
            zaxis=dict(title="z", tickvals=zt, ticktext=[str(t) for t in ticks],
                       range=[0.5, zt[-1] + 0.5], autorange=False),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=1.6 if mode == "exploded" else 1))

    hover_c = "x, y, z = %{customdata[0]}, %{customdata[1]}, %{customdata[2]}  step %{customdata[3]}<extra>cell</extra>"
    hover_w = "waypoint %{text}: x, y, z = %{customdata[0]}, %{customdata[1]}, %{customdata[2]}  step %{customdata[3]}<extra></extra>"
    hover_p = "step %{customdata[3]}: x, y, z = %{customdata[0]}, %{customdata[1]}, %{customdata[2]}<extra>path</extra>"
    cells, wps, line, walls = view("all")
    fig = go.Figure([
        go.Scatter3d(name="cells", mode="markers", x=cells["x"], y=cells["y"], z=cells["z"],
                     customdata=cells["cd"], hovertemplate=hover_c,
                     marker=dict(size=3, color="#9ca3af", opacity=0.35)),
        go.Scatter3d(name="waypoints", mode="markers+text", x=wps["x"], y=wps["y"], z=wps["z"],
                     text=wps["text"], customdata=wps["cd"], hovertemplate=hover_w,
                     marker=dict(size=9, color="#f59e0b", line=dict(color="#111827", width=1)),
                     textfont=dict(size=14, color="#111827", family="Arial Black, Arial, sans-serif")),
        go.Scatter3d(name="path", mode="lines", x=line["x"], y=line["y"], z=line["z"],
                     customdata=line["cd"], hovertemplate=hover_p, connectgaps=False,
                     visible=True if path else "legendonly",
                     line=dict(width=7, color=line["color"] or [0], colorscale="Viridis",
                               cmin=1, cmax=N, colorbar=dict(title="step", len=0.6) if path else None)),
        go.Scatter3d(name="walls", mode="lines", x=walls["x"], y=walls["y"], z=walls["z"],
                     hoverinfo="skip", line=dict(width=5, color="#dc2626"), connectgaps=False),
    ])

    def upd(key):
        mode = key if key in ("all", "exploded") else "slice"
        cells, wps, line, walls = view(mode, int(key) - 1 if mode == "slice" else None)
        data = {"x": [cells["x"], wps["x"], line["x"], walls["x"]],
                "y": [cells["y"], wps["y"], line["y"], walls["y"]],
                "z": [cells["z"], wps["z"], line["z"], walls["z"]],
                "customdata": [cells["cd"], wps["cd"], line["cd"], None],
                "text": [None, wps["text"], None, None],
                "line.color": [None, None, line["color"] or [0], "#dc2626"]}
        lay = {f"scene.{ax}.{k}": v for ax, d in scene(mode).items() if isinstance(d, dict)
               for k, v in d.items()}
        return [data, lay]

    steps = [dict(method="update", label=key, args=upd(key))
             for key in ["all", "exploded"] + [str(z + 1) for z in range(n)]]
    menus = []
    if path:
        menus.append(dict(type="buttons", direction="right", x=0.0, y=1.0, xanchor="left",
                          yanchor="bottom", showactive=True, active=0, buttons=[
                              dict(label="show solution", method="restyle", args=[{"visible": True}, [2]]),
                              dict(label="hide solution", method="restyle", args=[{"visible": False}, [2]])]))
    fig.update_layout(
        title=dict(text=f"3D Zip {n}³: {p.K} waypoints, {len(p.walls)} walls"
                   + (f", path {len(path)}/{N}" if path else ""), x=0.5, xanchor="center", y=0.98),
        scene=scene("all"), updatemenus=menus,
        sliders=[dict(active=0, currentvalue=dict(prefix="z-layer: "), steps=steps, pad=dict(t=30),
                      len=0.9, x=0.05)],
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=100, b=0))
    return fig
