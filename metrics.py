"""Evaluation metrics and an *independent* trajectory validator.

``IndependentValidator`` does not share code with ``collision_checker``: it
uses shapely polygons (footprint without safety margin) against the exact
squares of the occupied cells, on a trajectory densified to <= 2 cm. It is
used to re-verify every trajectory a planner reports as successful.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from occupancy_grid import OccupancyGrid
from trajectory import Trajectory
from vehicle import VehicleInfo, wrap_angle

try:
    import shapely
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    HAVE_SHAPELY = True
except Exception:  # pragma: no cover
    HAVE_SHAPELY = False


def densify(poses: np.ndarray, max_step: float = 0.02) -> np.ndarray:
    """Linear interpolation of (x, y, yaw) with proper yaw unwrapping."""
    if len(poses) < 2:
        return np.asarray(poses, dtype=float)
    out = [poses[:1]]
    for a, b in zip(poses[:-1], poses[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(d / max_step)))
        t = np.arange(1, n + 1)[:, None] / n
        dyaw = wrap_angle(b[2] - a[2])
        seg = np.hstack([a[:2] + t * (b[:2] - a[:2]), a[2] + t * dyaw])
        out.append(seg)
    res = np.vstack(out)
    res[:, 2] = (res[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return res


class IndependentValidator:
    def __init__(self, grid: OccupancyGrid, vehicle: VehicleInfo, margin: float = 0.0):
        self.grid = grid
        self.vehicle = vehicle
        self.margin = margin
        iy, ix = np.nonzero(grid.data)
        r = grid.resolution
        x0 = grid.origin[0] + ix * r
        y0 = grid.origin[1] + iy * r
        self._boxes = np.column_stack([x0, y0, x0 + r, y0 + r])
        if HAVE_SHAPELY and len(ix):
            self._geoms = shapely.box(self._boxes[:, 0], self._boxes[:, 1], self._boxes[:, 2], self._boxes[:, 3])
            self._tree = STRtree(self._geoms)
        else:
            self._geoms, self._tree = None, None

    def colliding_indices(self, poses: np.ndarray, densify_step: float = 0.02) -> List[int]:
        """Indices (into the densified pose array) of colliding poses."""
        dense = densify(np.asarray(poses, dtype=float), densify_step)
        xmin, xmax, ymin, ymax = self.grid.extent
        bad = []
        for i, p in enumerate(dense):
            fp = self.vehicle.footprint_world(p[0], p[1], p[2], self.margin)
            if (fp[:, 0].min() < xmin or fp[:, 0].max() > xmax or fp[:, 1].min() < ymin
                    or fp[:, 1].max() > ymax):
                bad.append(i)
                continue
            if self._tree is None:
                if not HAVE_SHAPELY and self._point_sample_hit(fp):
                    bad.append(i)
                continue
            poly = Polygon(fp)
            cand = self._tree.query(poly, predicate="intersects")
            if len(cand) and np.any(shapely.area(shapely.intersection(self._geoms[cand], poly)) > 1e-9):
                bad.append(i)
        return bad

    def _point_sample_hit(self, fp: np.ndarray) -> bool:  # fallback without shapely
        from matplotlib.path import Path
        xs = np.linspace(fp[:, 0].min(), fp[:, 0].max(), 60)
        ys = np.linspace(fp[:, 1].min(), fp[:, 1].max(), 60)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        inside = Path(fp).contains_points(pts)
        return bool(self.grid.is_occupied(pts[inside, 0], pts[inside, 1]).any())

    def is_collision_free(self, poses: np.ndarray) -> bool:
        return len(self.colliding_indices(poses)) == 0


def trajectory_metrics(traj: Optional[Trajectory], goal, vehicle: VehicleInfo, checker=None,
                       clearance_cap: float = 5.0, short_maneuver: float = 1.0) -> Dict[str, float]:
    """``short_maneuvers``: number of forward/reverse runs shorter than
    ``short_maneuver`` metres in a path with at least one gear change."""
    nan = float("nan")
    if traj is None or len(traj) == 0:
        return dict(trajectory_length=nan, forward_length=nan, reverse_length=nan, direction_changes=nan,
                    min_clearance=nan, max_curvature=nan, max_steering_deg=nan, goal_position_error=nan,
                    goal_yaw_error_deg=nan, n_points=0, min_maneuver_length=nan, short_maneuvers=nan,
                    min_road_maneuver=nan, min_slot_maneuver=nan, max_steer_reversals=nan, min_steer_hold=nan)
    poses = traj.poses()
    fwd, rev = traj.directional_lengths()
    end = traj.points[-1]
    clearance = nan
    if checker is not None:
        clearance = float(np.min(checker.min_clearance(poses, clearance_cap)))
    parts = traj.split_by_direction()
    runs = [p.points[-1].arc_length - p.points[0].arc_length for p in parts]
    # a manoeuvre whose gear change (its end, except the last one) is within half a
    # vehicle length of the goal is an in-slot correction; others are "road" moves
    zone = 0.5 * vehicle.vehicle_length
    road, slot = [], []
    for k, (p, r) in enumerate(zip(parts, runs)):
        q = p.points[-1] if k < len(parts) - 1 else p.points[0]
        (slot if math.hypot(q.x - goal[0], q.y - goal[1]) <= zone else road).append(r)
    reversals = 0
    for p in parts:
        sg = [1 if q.steering > 1e-6 else -1 for q in p.points[1:] if abs(q.steering) > 1e-6]
        reversals = max(reversals, sum(1 for a, b in zip(sg, sg[1:]) if a != b))
    # shortest steering hold: distance a steering sign is kept before it changes
    # (a few cm = a steering spike). The last segment of each manoeuvre may be cut short.
    holds = []
    for p in parts:
        cur, start = None, p.points[0].arc_length
        for q in p.points[1:]:
            s_ = 1 if q.steering > 1e-6 else (-1 if q.steering < -1e-6 else 0)
            if cur is not None and s_ != cur:
                holds.append(q.arc_length - start)
                start = q.arc_length
            if cur is None or s_ != cur:
                cur = s_
    multi = len(runs) > 1
    return dict(
        min_maneuver_length=float(min(runs)) if runs else nan,
        min_road_maneuver=float(min(road)) if multi and road else nan,
        min_slot_maneuver=float(min(slot)) if multi and slot else nan,
        max_steer_reversals=int(reversals),
        min_steer_hold=float(min(holds)) if holds else nan,
        short_maneuvers=int(sum(1 for r in runs if r < short_maneuver)) if len(runs) > 1 else 0,
        trajectory_length=traj.length,
        forward_length=fwd,
        reverse_length=rev,
        direction_changes=traj.n_direction_changes,
        min_clearance=clearance,
        max_curvature=float(np.max(np.abs(traj.curvatures()))),
        max_steering_deg=math.degrees(float(np.max(np.abs(traj.steerings())))),
        goal_position_error=math.hypot(end.x - goal[0], end.y - goal[1]),
        goal_yaw_error_deg=abs(math.degrees(wrap_angle(end.yaw - goal[2]))),
        n_points=len(traj),
    )
