"""Heuristics for the state-lattice search.

All heuristics are evaluated in batch: ``h(poses) -> (N,)`` with poses (N,3).

* ``EuclideanHeuristic``      : straight-line distance.
* ``EuclidYawHeuristic``      : max(euclid, R_min * |dyaw|) - yaw aware lower bound.
* ``ReedsSheppHeuristic``     : non-holonomic, obstacle-free lower bound. Uses a
  radius-normalised lookup table (one table for every vehicle) built once with
  the vectorised RS solver and cached on disk.
* ``ObstacleHeuristic``       : holonomic, obstacle-aware 2D Dijkstra from the
  goal on a coarse grid. Obstacles are inflated by a *provable* lower bound of
  the rear-axle clearance so the heuristic is (nearly) admissible and an
  infinite value proves unreachability.
* ``CombinedHeuristic``       : element-wise max of the above.
"""
from __future__ import annotations

import math
import os
import tempfile
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import _accel
from occupancy_grid import OccupancyGrid
from reeds_shepp import rs_length_vec
from vehicle import VehicleInfo

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def _relative_goal(poses: np.ndarray, goal):
    dx = goal[0] - poses[:, 0]
    dy = goal[1] - poses[:, 1]
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    x = c * dx + s * dy
    y = -s * dx + c * dy
    phi = (goal[2] - poses[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return x, y, phi


class Heuristic:
    name = "base"

    def __call__(self, poses: np.ndarray, directions=None, remaining=None) -> np.ndarray:
        raise NotImplementedError

    def single(self, x: float, y: float, yaw: float) -> float:
        return float(self(np.array([[x, y, yaw]]))[0])


class EuclideanHeuristic(Heuristic):
    name = "euclidean"

    def __init__(self, goal):
        self.goal = goal

    def __call__(self, poses, directions=None, remaining=None):
        return np.hypot(poses[:, 0] - self.goal[0], poses[:, 1] - self.goal[1])


class EuclidYawHeuristic(Heuristic):
    name = "euclid_yaw"

    def __init__(self, goal, min_radius: float):
        self.goal = goal
        self.r = min_radius

    def __call__(self, poses, directions=None, remaining=None):
        d = np.hypot(poses[:, 0] - self.goal[0], poses[:, 1] - self.goal[1])
        dyaw = np.abs((self.goal[2] - poses[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        return np.maximum(d, self.r * dyaw)


# ---------------------------------------------------------- RS lookup table
class _RSTable:
    """Shortest Reeds-Shepp length for a unit turning radius, tabulated over the
    goal pose relative to the vehicle (x in [-RANGE, RANGE], y >= 0 by
    reflection symmetry, 72 headings). One table serves every vehicle; it is
    built once and cached on disk."""

    STEP = 0.08
    RANGE = 6.0
    N_YAW = 72
    VERSION = 4
    _instance: Optional["_RSTable"] = None

    def __init__(self):
        self.n = int(round(self.RANGE / self.STEP)) + 1
        self.nx = 2 * self.n - 1           # x index i <-> x = (i - (n-1)) * STEP
        self.dyaw = 2 * math.pi / self.N_YAW
        path = os.path.join(_CACHE_DIR, f"rs_length_v{self.VERSION}_s{self.STEP}_R{self.RANGE}_y{self.N_YAW}.npy")
        table = None
        if os.path.exists(path):
            try:
                table = np.load(path)
                if table.shape != (self.nx, self.n, self.N_YAW):
                    table = None
            except Exception:
                table = None
        if table is None:
            xs = (np.arange(self.nx) - (self.n - 1)) * self.STEP
            gy, gp = np.meshgrid(np.arange(self.n) * self.STEP, -math.pi + np.arange(self.N_YAW) * self.dyaw,
                                 indexing="ij")
            table = np.stack([rs_length_vec(np.full(gy.shape, x), gy, gp) for x in xs]).astype(np.float32)
            try:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".npy")
                with os.fdopen(fd, "wb") as f:
                    np.save(f, table)
                os.replace(tmp, path)  # atomic: safe with parallel workers
            except OSError:
                pass
        self.table = table

    @classmethod
    def get(cls) -> "_RSTable":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def lookup(self, x, y, phi):
        """Trilinear lookup (unit radius). NaN outside the table."""
        if _accel.HAVE_NUMBA:
            return _accel.rs_table_lookup(self.table,
                                          np.ascontiguousarray(x, dtype=np.float64) + (self.n - 1) * self.STEP,
                                          np.ascontiguousarray(y, dtype=np.float64),
                                          np.ascontiguousarray(phi, dtype=np.float64), self.STEP, self.dyaw)
        return self.lookup_numpy(x, y, phi)

    def lookup_numpy(self, x, y, phi):
        neg_y = y < 0
        y = np.abs(y)
        phi = np.where(neg_y, -phi, phi)
        fx = x / self.STEP + (self.n - 1)
        fy = y / self.STEP
        inside = (fx >= 0) & (fx < self.nx - 1) & (fy < self.n - 1)
        fx = np.clip(fx, 0.0, self.nx - 1.001)
        fy = np.minimum(fy, self.n - 1.001)
        fp = ((phi + math.pi) % (2 * math.pi)) / self.dyaw
        ix, iy, ip = fx.astype(np.int64), fy.astype(np.int64), fp.astype(np.int64) % self.N_YAW
        tx, ty, tp = fx - ix, fy - iy, fp - np.floor(fp)
        ip1 = (ip + 1) % self.N_YAW
        t = self.table
        c00 = t[ix, iy, ip] * (1 - tp) + t[ix, iy, ip1] * tp
        c10 = t[ix + 1, iy, ip] * (1 - tp) + t[ix + 1, iy, ip1] * tp
        c01 = t[ix, iy + 1, ip] * (1 - tp) + t[ix, iy + 1, ip1] * tp
        c11 = t[ix + 1, iy + 1, ip] * (1 - tp) + t[ix + 1, iy + 1, ip1] * tp
        v = (c00 * (1 - tx) + c10 * tx) * (1 - ty) + (c01 * (1 - tx) + c11 * tx) * ty
        return np.where(inside, v, np.nan)


class ReedsSheppHeuristic(Heuristic):
    """Shortest Reeds-Shepp length to the goal (obstacle-free lower bound)."""

    name = "reeds_shepp"

    def __init__(self, goal, min_radius: float):
        self.goal = goal
        self.r = min_radius
        self.table = _RSTable.get()
        self.slack = self.table.STEP * min_radius  # interpolation slack (stays optimistic)

    def __call__(self, poses, directions=None, remaining=None):
        x, y, phi = _relative_goal(poses, self.goal)
        v = self.table.lookup(x / self.r, y / self.r, phi) * self.r
        far = np.isnan(v)
        if far.any():
            d = np.hypot(x[far], y[far])
            v[far] = np.maximum(d, self.r * np.abs(phi[far]))
        return np.maximum(v - self.slack, 0.0)


# ------------------------------------------------------- obstacle heuristic
class ObstacleHeuristic(Heuristic):
    """2D Dijkstra distance-to-goal of the rear-axle point."""

    name = "obstacle"

    def __init__(self, grid: OccupancyGrid, goal, vehicle: VehicleInfo, margin: float,
                 edt: np.ndarray, resolution: float = 0.2, start=None, point_robot: bool = False):
        self.goal = goal
        res = max(resolution, grid.resolution)
        self.res = res
        xmin, xmax, ymin, ymax = grid.extent
        self.origin = (xmin, ymin)
        self.nx = max(1, int(math.ceil((xmax - xmin) / res)))
        self.ny = max(1, int(math.ceil((ymax - ymin) / res)))
        cx = xmin + (np.arange(self.nx) + 0.5) * res
        cy = ymin + (np.arange(self.ny) + 0.5) * res
        gx, gy = np.meshgrid(cx, cy)
        # --- soundness argument (a blocked cell must be provably unusable) ---------
        # Every collision-free pose (checked with ``margin``) has a disk of radius
        # r0 around the rear axle that contains no occupied cell square.  Samples of
        # a path are <= sample_ds apart (< res), so the coarse cells containing them
        # form an 8-connected chain, and every such cell centre c lies within
        # res/sqrt(2) of a sample: d_true(c) >= r0 - res/sqrt(2).
        # An UPPER bound of d_true(c) from the fine distance field is
        #   ub(c) = edt(f) + |c - f|    (f = centre of the fine cell containing c)
        # so blocking cells with ub(c) < r0 - res/sqrt(2) never blocks a cell that
        # a feasible path can visit -> an infinite value proves unreachability.
        fix = np.clip(np.floor((gx - grid.origin[0]) / grid.resolution).astype(np.int64), 0, grid.width - 1)
        fiy = np.clip(np.floor((gy - grid.origin[1]) / grid.resolution).astype(np.int64), 0, grid.height - 1)
        fcx = grid.origin[0] + (fix + 0.5) * grid.resolution
        fcy = grid.origin[1] + (fiy + 0.5) * grid.resolution
        ub = edt[fiy, fix] + np.hypot(gx - fcx, gy - fcy)
        if point_robot:
            r0 = 0.0  # a point robot has no footprint disk: only occupied cells are blocked
        else:
            r0 = min(vehicle.rear_overhang, vehicle.wheel_base + vehicle.front_overhang,
                     vehicle.wheel_tread / 2 + vehicle.left_overhang,
                     vehicle.wheel_tread / 2 + vehicle.right_overhang) + margin
        self.inflation = max(0.0, r0 - res / math.sqrt(2.0) - 1e-6)
        free = ub >= self.inflation
        # map border: the whole (inflated) footprint is inside the map, so the rear
        # axle is >= r0 from every map edge; same res/sqrt(2) slack for the centre
        border = self.inflation
        free &= (gx - xmin >= border) & (xmax - gx >= border) & (gy - ymin >= border) & (ymax - gy >= border)
        gix, giy = self._index(goal[0], goal[1])
        self.goal_valid = 0 <= gix < self.nx and 0 <= giy < self.ny
        if self.goal_valid:
            free[giy, gix] = True
        self.free = free
        self.dist = np.full((self.ny, self.nx), np.inf)
        if self.goal_valid:
            self.dist = self._dijkstra(free, giy * self.nx + gix).reshape(self.ny, self.nx)
        self.offset = 2.0 * res  # slack for discretisation (keeps it optimistic)

    def _index(self, x, y):
        return (int(math.floor((x - self.origin[0]) / self.res)),
                int(math.floor((y - self.origin[1]) / self.res)))

    def _dijkstra(self, free: np.ndarray, source: int) -> np.ndarray:
        ny, nx = free.shape
        idx = np.arange(nx * ny).reshape(ny, nx)
        rows, cols, w = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ys0, ys1 = max(0, -dy), ny - max(0, dy)
            xs0, xs1 = max(0, -dx), nx - max(0, dx)
            a = free[ys0:ys1, xs0:xs1]
            b = free[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            ok = a & b
            ia = idx[ys0:ys1, xs0:xs1][ok]
            ib = idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][ok]
            cost = self.res * math.hypot(dx, dy)
            rows += [ia, ib]
            cols += [ib, ia]
            w += [np.full(ia.size, cost), np.full(ia.size, cost)]
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        w = np.concatenate(w)
        g = csr_matrix((w, (rows, cols)), shape=(nx * ny, nx * ny))
        return dijkstra(g, directed=False, indices=source)

    def __call__(self, poses, directions=None, remaining=None):
        ix = np.floor((poses[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((poses[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        out = np.full(len(poses), np.inf)
        out[inside] = self.dist[iy[inside], ix[inside]]
        return np.maximum(out - self.offset, 0.0)

    def reachable(self, x: float, y: float) -> bool:
        return bool(np.isfinite(self(np.array([[x, y, 0.0]]))[0]))


class LatticeHeuristic(Heuristic):
    """Free-space lattice cost-to-go (see lattice_heuristic.py); outside the
    table it falls back to the given heuristic (cost-aware RS)."""

    name = "lattice"

    def __init__(self, goal, hlut, fallback: Heuristic):
        self.goal = goal
        self.hlut = hlut
        self.fallback = fallback
        self.cg, self.sg = math.cos(goal[2]), math.sin(goal[2])

    def __call__(self, poses, directions=None, remaining=None):
        dx = poses[:, 0] - self.goal[0]
        dy = poses[:, 1] - self.goal[1]
        x = self.cg * dx + self.sg * dy
        y = -self.sg * dx + self.cg * dy
        phi = poses[:, 2] - self.goal[2]
        d = np.zeros(len(poses), dtype=np.int64) if directions is None else np.asarray(directions)
        v = self.hlut.lookup(x, y, phi, d, remaining)
        miss = np.isnan(v)
        if miss.any():
            v[miss] = self.fallback(poses[miss], None if directions is None else d[miss])
        return v


class CombinedHeuristic(Heuristic):
    name = "combined"

    def __init__(self, parts: Sequence[Heuristic]):
        self.parts = list(parts)

    def __call__(self, poses, directions=None, remaining=None):
        h = self.parts[0](poses, directions, remaining)
        for p in self.parts[1:]:
            h = np.maximum(h, p(poses, directions, remaining))
        return h


HEURISTIC_MODES = ("euclidean", "euclid_yaw", "reeds_shepp", "obstacle", "rs_obstacle", "lattice_obstacle")


def build_heuristic(mode: str, grid: OccupancyGrid, goal, vehicle: VehicleInfo, margin: float,
                    edt: np.ndarray, obstacle_resolution: float = 0.2,
                    hlut=None, point_robot: bool = False):
    """Returns (heuristic, obstacle_heuristic_or_None)."""
    r = vehicle.min_turning_radius
    if mode == "euclidean":
        return EuclideanHeuristic(goal), None
    if mode == "euclid_yaw":
        return EuclidYawHeuristic(goal, r), None
    if mode == "reeds_shepp":
        return ReedsSheppHeuristic(goal, r), None
    obs = ObstacleHeuristic(grid, goal, vehicle, margin, edt, obstacle_resolution, point_robot=point_robot)
    if mode == "obstacle":
        return CombinedHeuristic([obs, EuclidYawHeuristic(goal, r)]), obs
    if mode == "rs_obstacle" or (mode == "lattice_obstacle" and hlut is None):
        return CombinedHeuristic([ReedsSheppHeuristic(goal, r), obs]), obs
    if mode == "lattice_obstacle":
        lat = LatticeHeuristic(goal, hlut, ReedsSheppHeuristic(goal, r))
        return CombinedHeuristic([lat, obs]), obs
    raise ValueError(f"unknown heuristic mode {mode!r}")
