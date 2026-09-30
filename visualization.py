"""matplotlib visualisation of a planning result."""
from __future__ import annotations

import math
from typing import Optional, Sequence


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from vehicle import VehicleInfo

# reference palette (categorical slots in fixed order)
C_FORWARD = "#2a78d6"   # slot 1 blue
C_REVERSE = "#eb6834"   # slot 2 orange
C_START = "#1baf7a"     # slot 3 aqua
C_GOAL = "#4a3aa7"      # slot 7 violet
C_SWITCH = "#e34948"    # slot 8 red
C_EXPLORED = "#b9b8b2"
C_GRID = "#3d3c39"
KIND_COLORS = {"vehicle": "#8fa3bf", "wall": "#52514e", "pillar": "#6b6a66", "obstacle": "#77766f",
               "curb": "#9d9c95"}


def _draw_footprint(ax, vehicle: VehicleInfo, pose, color, lw=1.0, ls="-", alpha=1.0, fill=False, z=4):
    fp = vehicle.footprint_world(pose[0], pose[1], pose[2])
    ax.add_patch(MplPolygon(fp, closed=True, fill=fill, facecolor=color if fill else "none",
                            edgecolor=color, lw=lw, ls=ls, alpha=alpha, zorder=z))
    L = vehicle.wheel_base * 0.6
    ax.annotate("", xy=(pose[0] + L * math.cos(pose[2]), pose[1] + L * math.sin(pose[2])),
                xytext=(pose[0], pose[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=alpha), zorder=z + 1)


def plot_result(grid, vehicle: VehicleInfo, start, goal, result=None, obstacles: Sequence = (),
                title: str = "", metrics: Optional[dict] = None, ax=None, footprint_step: float = 1.0,
                show_explored: bool = True):
    own = ax is None
    if own:
        xmin, xmax, ymin, ymax = grid.extent
        aspect = (ymax - ymin) / max(xmax - xmin, 1e-6)
        fig, ax = plt.subplots(figsize=(11, max(4.5, 11 * aspect + 1.2)))
    else:
        fig = ax.figure
    ax.imshow(grid.data, origin="lower", extent=grid.extent, cmap="Greys", vmin=0, vmax=1.6,
              interpolation="nearest", zorder=0)
    for ob in obstacles:
        kind = getattr(ob, "kind", "obstacle")
        ax.add_patch(MplPolygon(ob.polygon, closed=True, facecolor=KIND_COLORS.get(kind, "#77766f"),
                                edgecolor="white", lw=0.8, alpha=0.9, zorder=1))
    traj = result.trajectory if result is not None else None
    if show_explored and result is not None and result.explored is not None and len(result.explored):
        e = result.explored
        ax.scatter(e[:, 0], e[:, 1], s=1.2, c=C_EXPLORED, alpha=0.5, lw=0, zorder=2)

    _draw_footprint(ax, vehicle, start, C_START, lw=2.0, z=6)
    _draw_footprint(ax, vehicle, goal, C_GOAL, lw=2.0, ls="--", z=6)

    if traj is not None and len(traj) > 1:
        p = traj.poses()
        d = traj.directions()
        # footprints along the path
        s = np.array([pt.arc_length for pt in traj.points])
        marks = np.searchsorted(s, np.arange(0, s[-1], footprint_step))
        for i in marks:
            _draw_footprint(ax, vehicle, p[i], C_FORWARD if d[i] > 0 else C_REVERSE, lw=0.6, alpha=0.35, z=3)
        for i in range(1, len(p)):
            ax.plot(p[i - 1:i + 1, 0], p[i - 1:i + 1, 1], color=C_FORWARD if d[i] > 0 else C_REVERSE,
                    lw=2.2, solid_capstyle="round", zorder=5)
        for i in traj.switch_indices():
            ax.scatter([p[i, 0]], [p[i, 1]], s=90, marker="X", color=C_SWITCH, edgecolor="white", lw=1.0, zorder=7)

    handles = [Line2D([], [], color=C_FORWARD, lw=2.2, label="forward"),
               Line2D([], [], color=C_REVERSE, lw=2.2, label="reverse"),
               Line2D([], [], color=C_SWITCH, marker="X", ls="", ms=9, label="direction switch"),
               Line2D([], [], color=C_START, lw=2, label="start"),
               Line2D([], [], color=C_GOAL, lw=2, ls="--", label="goal"),
               Line2D([], [], color=C_EXPLORED, marker=".", ls="", ms=6, label="explored nodes")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(grid.extent[0], grid.extent[1])
    ax.set_ylim(grid.extent[2], grid.extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, color="#e1e0d9", lw=0.5, zorder=0)
    ax.tick_params(colors="#52514e", labelsize=8)
    txt = title
    if result is not None:
        txt += f"\nstatus={result.status.value}  time={result.planning_time * 1e3:.0f} ms  nodes={result.expanded_nodes}"
    if metrics and metrics.get("success"):
        txt += (f"  L={metrics['trajectory_length']:.1f} m  switches={metrics['direction_changes']}"
                f"  clearance={metrics['min_clearance']:.2f} m")
    ax.set_title(txt, fontsize=10, loc="left", color="#0b0b0b")
    if own:
        fig.tight_layout()
    return fig, ax


def _gif_axes(grid, vehicle, start, goal, obstacles, points, margin, title):
    """Scene figure cropped to ``points`` (+ start/goal footprints) for animations."""
    fp = np.vstack([vehicle.footprint_world(*q) for q in (start, goal)] + [np.asarray(points)[:, :2]])
    lo = np.maximum(fp.min(axis=0) - margin, [grid.extent[0], grid.extent[2]])
    hi = np.minimum(fp.max(axis=0) + margin, [grid.extent[1], grid.extent[3]])
    fig, ax = plt.subplots(figsize=(7, max(3.5, 7 * (hi[1] - lo[1]) / (hi[0] - lo[0]))), dpi=80)
    plot_result(grid, vehicle, start, goal, None, obstacles, ax=ax)
    ax.get_legend().remove()
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_title(title, fontsize=10, loc="left")
    label = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10, va="top", zorder=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    fig.tight_layout()
    return fig, ax, label


def _grab(fig):
    from PIL import Image
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(
        colors=64, method=0, dither=0)  # median cut, no dithering (clean flat colours)


def _save_gif(frames, path, fps, hold):
    frames = frames + [frames[-1]] * int(hold * fps)          # hold the last frame
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                   optimize=True)


def _split_by_gear(p, d):
    """(forward, reverse) polylines with NaN gaps; point i carries the motion arriving at it."""
    fwd = np.where(d[:, None] > 0, p[:, :2], np.nan)
    rev = np.where(d[:, None] < 0, p[:, :2], np.nan)
    for q, m in ((fwd, d > 0), (rev, d < 0)):               # join each run to its previous point
        start = np.nonzero(m[1:] & ~m[:-1])[0]
        q[start] = p[start, :2]
    return fwd, rev


def animate_result(grid, vehicle: VehicleInfo, start, goal, result, obstacles: Sequence = (), path: str = "parking.gif",
                   title: str = "", fps: int = 20, speed: float = 3.0, margin: float = 2.0,
                   footprint_step: float = 0.5):
    """GIF of the vehicle driving the planned trajectory (``speed`` m of path per
    second); the swept footprint is left behind every ``footprint_step`` m."""
    traj = result.trajectory
    p, d = traj.poses(), traj.directions()
    s = np.array([pt.arc_length for pt in traj.points])
    fig, ax, label = _gif_axes(grid, vehicle, start, goal, obstacles,
                               np.vstack([vehicle.footprint_world(*q) for q in p[::10]]), margin, title)
    fwd, rev = _split_by_gear(p, d)
    for q, c in ((fwd, C_FORWARD), (rev, C_REVERSE)):          # the whole plan, faint
        ax.plot(q[:, 0], q[:, 1], color=c, lw=1.2, alpha=0.35, zorder=3)
    trails = [ax.plot([], [], color=c, lw=2.2, zorder=6)[0] for c in (C_FORWARD, C_REVERSE)]
    car = MplPolygon(vehicle.footprint_world(*p[0]), closed=True, alpha=0.85, zorder=8)
    ax.add_patch(car)
    marks = list(np.searchsorted(s, np.arange(0.0, s[-1], footprint_step)))
    switches = np.array(traj.switch_indices(), dtype=int)
    n = int(math.ceil(s[-1] / speed * fps))
    frames = []
    for k in range(n + 1):
        i = min(int(np.searchsorted(s, s[-1] * k / max(n, 1))), len(p) - 1)
        while marks and marks[0] <= i:                          # footprint trace
            j = marks.pop(0)
            c = C_FORWARD if d[j] > 0 else C_REVERSE
            ax.add_patch(MplPolygon(vehicle.footprint_world(*p[j]), closed=True, fill=False,
                                    edgecolor=c, lw=0.7, alpha=0.45, zorder=4))
        car.set_xy(vehicle.footprint_world(*p[i]))
        car.set_facecolor(C_FORWARD if d[i] > 0 else C_REVERSE)
        for t, q in zip(trails, (fwd, rev)):
            t.set_data(q[:i + 1, 0], q[:i + 1, 1])
        label.set_text(f"{'forward' if d[i] > 0 else 'reverse'}   gear changes: {int((switches <= i).sum())}")
        frames.append(_grab(fig))
    plt.close(fig)
    _save_gif(frames, path, fps, hold=1.0)


def animate_search(grid, vehicle: VehicleInfo, start, goal, bres, obstacles: Sequence = (),
                   path: str = "search.gif", title: str = "", fps: int = 20, n_frames: int = 60,
                   chunk: int = 64, margin: float = 2.0):
    """GIF of the bidirectional search (``BidirectionalResult``): expanded nodes of
    the forward tree (start -> goal) and the exit tree (goal -> start) in
    expansion order (the searches alternate in chunks), then the returned path."""
    trees = [r.explored if r is not None and r.explored is not None else np.zeros((0, 2))
             for r in (bres.forward, bres.exit)]
    order = []                                                   # (tree, index) in expansion order
    pos = [0, 0]
    while pos[0] < len(trees[0]) or pos[1] < len(trees[1]):
        for t in (0, 1):
            m = min(chunk, len(trees[t]) - pos[t])
            order += [(t, pos[t] + i) for i in range(m)]
            pos[t] += m
    traj = bres.result.trajectory
    p, d = traj.poses(), traj.directions()
    fig, ax, label = _gif_axes(grid, vehicle, start, goal, obstacles, np.vstack([p[:, :2]] + trees), margin, title)
    colors = (C_START, C_GOAL)
    dots = [ax.scatter([], [], s=4, c=c, lw=0, alpha=0.7, zorder=3) for c in colors]
    ax.legend(handles=[Line2D([], [], color=c, marker=".", ls="", ms=7, label=l)
                       for c, l in zip(colors, ("forward tree (from start)", "exit tree (from goal)"))],
              loc="upper right", fontsize=8, framealpha=0.9)
    frames = []
    for k in range(1, n_frames + 1):
        upto = order[:int(len(order) * k / n_frames)]
        for t in (0, 1):
            idx = [i for tt, i in upto if tt == t]
            dots[t].set_offsets(trees[t][idx] if idx else np.zeros((0, 2)))
        n0 = sum(1 for tt, _ in upto if tt == 0)
        label.set_text(f"expanded  forward: {n0}   exit: {len(upto) - n0}")
        frames.append(_grab(fig))
    fwd, rev = _split_by_gear(p, d)
    sarc = np.array([pt.arc_length for pt in traj.points])
    for i in np.searchsorted(sarc, np.arange(0.0, sarc[-1], 1.0)):
        ax.add_patch(MplPolygon(vehicle.footprint_world(*p[i]), closed=True, fill=False,
                                edgecolor=C_FORWARD if d[i] > 0 else C_REVERSE, lw=0.7, alpha=0.5, zorder=4))
    for q, c in ((fwd, C_FORWARD), (rev, C_REVERSE)):
        ax.plot(q[:, 0], q[:, 1], color=c, lw=2.4, zorder=6)
    for i in traj.switch_indices():
        ax.scatter([p[i, 0]], [p[i, 1]], s=90, marker="X", color=C_SWITCH, edgecolor="white", lw=1.0, zorder=7)
    label.set_text(f"path found: {traj.n_direction_changes} gear change(s), {len(order)} expansions")
    frames.append(_grab(fig))
    plt.close(fig)
    _save_gif(frames, path, fps, hold=2.0)
