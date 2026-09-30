"""Vehicle footprint collision checking against an occupancy grid.

Two footprint models are available (``method``):

* ``"rectangle"`` (default, exact): oriented-rectangle vs occupied grid-cell
  squares using the separating axis theorem. A cascade of distance-field tests
  resolves most poses in O(1) and only ambiguous poses reach the exact test:

  1. map bounds (all footprint corners inside the map)
  2. circumscribed circle vs lower-bound distance      -> definitely free
  3. inscribed spine circles vs upper-bound distance   -> definitely colliding
  4. covering circles vs lower-bound distance          -> definitely free
  5. exact SAT against *boundary* occupied cells (KD-tree query)

* ``"circles"`` (approximate, conservative): the rectangle is covered by
  ``n_circles`` circles and each circle is tested against the distance field.
  It never misses a collision but rejects some feasible poses.

``margin`` inflates the footprint (safety distance): one value for every side or
(longitudinal, lateral). ``self.margin`` is the smaller of the two.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import _accel
from occupancy_grid import OccupancyGrid
from vehicle import VehicleInfo, split_margin

_BIG = 1.0e6


class CollisionChecker:
    METHODS = ("rectangle", "circles", "point")

    def __init__(self, grid: OccupancyGrid, vehicle: VehicleInfo, method: str = "rectangle",
                 margin=0.1, n_circles: int = 4, use_numba: Optional[bool] = None):
        if method not in self.METHODS:
            raise ValueError(f"unknown collision method {method!r}")
        m_lon, m_lat = split_margin(margin)
        if m_lon < 0 or m_lat < 0:
            raise ValueError("margin must be >= 0")
        self.grid = grid
        self.vehicle = vehicle
        self.method = method
        self.margin_lon, self.margin_lat = m_lon, m_lat
        self.margin = min(m_lon, m_lat)
        self.n_circles = int(n_circles)
        self.check_count = 0          # number of poses checked
        self.exact_count = 0          # number of poses resolved by exact SAT
        self.use_numba = _accel.HAVE_NUMBA if use_numba is None else (use_numba and _accel.HAVE_NUMBA)

        r = grid.resolution
        self._res = r
        self._h = r / 2.0
        occ = grid.data > 0
        self._has_obstacle = bool(occ.any())
        if self._has_obstacle:
            self.edt = ndimage.distance_transform_edt(~occ) * r
        else:
            self.edt = np.full(occ.shape, _BIG)
        # boundary occupied cells (4-neighbour adjacent to free space)
        if self._has_obstacle:
            eroded = ndimage.binary_erosion(occ, structure=ndimage.generate_binary_structure(2, 1),
                                            border_value=1)
            boundary = occ & ~eroded
            iy, ix = np.nonzero(boundary)
            bx, by = grid.index_to_world(ix, iy)
            self._bcells = np.column_stack([bx, by])
            self._tree: Optional[cKDTree] = cKDTree(self._bcells) if len(bx) else None
        else:
            self._bcells = np.zeros((0, 2))
            self._tree = None

        v = vehicle
        self._cx, self._cy = v.footprint_center_offset
        self._a0, self._b0 = v.half_length, v.half_width
        self._a = self._a0 + self.margin_lon
        self._b = self._b0 + self.margin_lat
        self._rcirc = math.hypot(self._a, self._b)
        # inscribed spine circles (radius b) -> definite collision test
        span = max(self._a - self._b, 0.0)
        k = max(2, int(math.ceil(2 * span / max(self._b, 1e-3))) + 1)
        self._spine_x = self._cx + np.linspace(-span, span, k)
        self._spine_r = min(self._a, self._b)
        # covering circles for the free-space filter / circle method
        self._cover_x, self._cover_r = self._covering_circles(8)
        self._circ_x, self._circ_r = self._covering_circles(self.n_circles)
        self._data_c = np.ascontiguousarray(grid.data)
        self._edt_c = np.ascontiguousarray(self.edt, dtype=np.float64)

    # --------------------------------------------------------------- helpers
    def _covering_circles(self, n: int):
        n = max(1, n)
        seg = 2.0 * self._a / n
        xs = self._cx - self._a + seg * (np.arange(n) + 0.5)
        return xs, math.hypot(seg / 2.0, self._b)

    def _local_points(self, poses: np.ndarray, lx: np.ndarray, ly: float):
        """World coordinates of local points (lx[k], ly) for every pose -> (N,K,2)."""
        c = np.cos(poses[:, 2])[:, None]
        s = np.sin(poses[:, 2])[:, None]
        lx = np.asarray(lx)[None, :]
        wx = poses[:, 0:1] + c * lx - s * ly
        wy = poses[:, 1:2] + s * lx + c * ly
        return wx, wy

    def _cell_lookup(self, wx: np.ndarray, wy: np.ndarray):
        g = self.grid
        ix = np.floor((wx - g.origin[0]) / self._res).astype(np.int64)
        iy = np.floor((wy - g.origin[1]) / self._res).astype(np.int64)
        inside = (ix >= 0) & (ix < g.width) & (iy >= 0) & (iy < g.height)
        ixc = np.clip(ix, 0, g.width - 1)
        iyc = np.clip(iy, 0, g.height - 1)
        d = self.edt[iyc, ixc]
        ccx = g.origin[0] + (ixc + 0.5) * self._res
        ccy = g.origin[1] + (iyc + 0.5) * self._res
        off = np.hypot(wx - ccx, wy - ccy)
        return d, off, inside

    def distance_lower_bound(self, wx, wy):
        """Lower bound of the distance from points to the occupied cell squares."""
        d, off, inside = self._cell_lookup(wx, wy)
        lb = d - off - self._h * math.sqrt(2.0)
        return np.where(inside, lb, -_BIG)

    def distance_upper_bound(self, wx, wy):
        d, off, inside = self._cell_lookup(wx, wy)
        return np.where(inside, d + off, _BIG)

    def distance_estimate(self, wx, wy):
        d, _, inside = self._cell_lookup(wx, wy)
        return np.where(inside, np.maximum(d - self._h, 0.0), 0.0)

    def footprint_corners(self, poses: np.ndarray, margin=None) -> np.ndarray:
        m_lon, m_lat = (self.margin_lon, self.margin_lat) if margin is None else split_margin(margin)
        a, b = self._a0 + m_lon, self._b0 + m_lat
        lx = np.array([self._cx + a, self._cx + a, self._cx - a, self._cx - a])
        ly = np.array([self._cy - b, self._cy + b, self._cy + b, self._cy - b])
        c = np.cos(poses[:, 2])[:, None]
        s = np.sin(poses[:, 2])[:, None]
        wx = poses[:, 0:1] + c * lx[None, :] - s * ly[None, :]
        wy = poses[:, 1:2] + s * lx[None, :] + c * ly[None, :]
        return np.stack([wx, wy], axis=-1)

    def _in_map(self, poses: np.ndarray) -> np.ndarray:
        corners = self.footprint_corners(poses)
        xmin, xmax, ymin, ymax = self.grid.extent
        return np.all((corners[..., 0] >= xmin) & (corners[..., 0] < xmax) &
                      (corners[..., 1] >= ymin) & (corners[..., 1] < ymax), axis=1)

    # ------------------------------------------------------------ main API
    def check_pose(self, x: float, y: float, yaw: float) -> bool:
        """True if the pose collides."""
        return bool(self.check_poses(np.array([[x, y, yaw]], dtype=float))[0])

    def check_poses(self, poses: np.ndarray) -> np.ndarray:
        """Vectorised collision test. Returns bool array (True = collision)."""
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        n = len(poses)
        self.check_count += n
        if n == 0:
            return np.zeros(0, dtype=bool)
        if self.method == "point":
            return self.grid.is_occupied(poses[:, 0], poses[:, 1])
        if self.use_numba:
            out, n_exact = _accel.check_rect_batch(
                np.ascontiguousarray(poses), self._data_c, self._edt_c, self.grid.origin[0],
                self.grid.origin[1], self._res, self._cx, self._cy, self._a, self._b,
                self._spine_x, self._spine_r, self._cover_x, self._cover_r, self._rcirc,
                self.method == "circles", self._circ_x, self._circ_r)
            self.exact_count += int(n_exact)
            return out
        return self._check_numpy(poses)

    def any_collision(self, poses: np.ndarray) -> bool:
        """True if any pose collides; stops at the first hit (fast rejection)."""
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        if len(poses) == 0:
            return False
        if self.use_numba and self.method != "point":
            out, n_exact = _accel.check_rect_batch(
                np.ascontiguousarray(poses), self._data_c, self._edt_c, self.grid.origin[0],
                self.grid.origin[1], self._res, self._cx, self._cy, self._a, self._b,
                self._spine_x, self._spine_r, self._cover_x, self._cover_r, self._rcirc,
                self.method == "circles", self._circ_x, self._circ_r, True)
            k = int(np.argmax(out)) + 1 if out.any() else len(poses)
            self.check_count += k
            self.exact_count += int(n_exact)
            return bool(out.any())
        return bool(self.check_poses(poses).any())

    def _check_numpy(self, poses: np.ndarray) -> np.ndarray:
        """Reference implementation of the cascade (pure numpy)."""
        bad = ~np.all(np.isfinite(poses), axis=1)          # invalid pose: never free
        poses = np.where(bad[:, None], 0.0, poses)
        collide = ~self._in_map(poses) | bad
        if not self._has_obstacle:
            return collide
        if self.method == "circles":
            wx, wy = self._local_points(poses, self._circ_x, self._cy)
            lb = self.distance_lower_bound(wx, wy)
            return collide | np.any(lb < self._circ_r, axis=1)

        # rectangle cascade
        todo = ~collide
        cwx, cwy = self._local_points(poses, np.array([self._cx]), self._cy)
        lb_c = self.distance_lower_bound(cwx, cwy)[:, 0]
        todo &= ~(lb_c > self._rcirc)
        if not todo.any():
            return collide
        idx = np.nonzero(todo)[0]
        sub = poses[idx]
        swx, swy = self._local_points(sub, self._spine_x, self._cy)
        sure_hit = np.any(self.distance_upper_bound(swx, swy) < self._spine_r, axis=1)
        # a spine point inside an occupied cell is a collision; needed because the
        # exact stage below only looks at *boundary* cells (footprint fully inside
        # a thick obstacle with coarse cells would otherwise be missed)
        sure_hit |= np.any(self.grid.is_occupied(swx, swy), axis=1)
        collide[idx[sure_hit]] = True
        idx, sub = idx[~sure_hit], sub[~sure_hit]
        if len(idx) == 0:
            return collide
        vwx, vwy = self._local_points(sub, self._cover_x, self._cy)
        sure_free = np.all(self.distance_lower_bound(vwx, vwy) > self._cover_r, axis=1)
        idx, sub = idx[~sure_free], sub[~sure_free]
        if len(idx) == 0:
            return collide
        collide[idx] = self._exact(sub)
        return collide

    def _exact(self, poses: np.ndarray, a: Optional[float] = None, b: Optional[float] = None) -> np.ndarray:
        """Exact oriented rectangle vs boundary-cell squares (SAT)."""
        self.exact_count += len(poses)
        if self._tree is None:
            return np.zeros(len(poses), dtype=bool)
        a = self._a if a is None else a
        b = self._b if b is None else b
        h = self._h
        c = np.cos(poses[:, 2])
        s = np.sin(poses[:, 2])
        cx = poses[:, 0] + c * self._cx - s * self._cy
        cy = poses[:, 1] + s * self._cx + c * self._cy
        radius = math.hypot(a, b) + h * math.sqrt(2.0) + 1e-9
        lists = self._tree.query_ball_point(np.column_stack([cx, cy]), radius)
        lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
        out = np.zeros(len(poses), dtype=bool)
        if lens.sum() == 0:
            return out
        rep = np.repeat(np.arange(len(poses)), lens)
        cells = self._bcells[np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if l])]
        dx = cells[:, 0] - cx[rep]
        dy = cells[:, 1] - cy[rep]
        ux, uy = c[rep], s[rep]
        aux, auy = np.abs(ux), np.abs(uy)
        sep = np.abs(dx) > h + a * aux + b * auy
        sep |= np.abs(dy) > h + a * auy + b * aux
        sep |= np.abs(dx * ux + dy * uy) > a + h * (aux + auy)
        sep |= np.abs(-dx * uy + dy * ux) > b + h * (aux + auy)
        hit = ~sep
        np.logical_or.at(out, rep[hit], True)
        return out

    # ------------------------------------------------------------ clearance
    def clearance_estimate(self, poses: np.ndarray) -> np.ndarray:
        """Cheap approximate body clearance (without margin) used in the cost."""
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        if not self._has_obstacle:
            return np.full(len(poses), _BIG)
        if self.use_numba:
            return _accel.clearance_estimate_batch(np.ascontiguousarray(poses), self._edt_c, self.grid.origin[0],
                                                   self.grid.origin[1], self._res, self._cy, self._spine_x,
                                                   self._b0)
        wx, wy = self._local_points(poses, self._spine_x, self._cy)
        d = self.distance_estimate(wx, wy)
        return np.maximum(np.min(d, axis=1) - self._b0, 0.0)

    def min_clearance(self, poses: np.ndarray, cap: float = 5.0) -> np.ndarray:
        """Exact distance from the bare footprint rectangle to the nearest occupied
        cell square, capped at ``cap``."""
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        out = np.full(len(poses), cap)
        if self._tree is None or len(poses) == 0:
            return out
        a, b, h = self._a0, self._b0, self._h
        if self.use_numba:
            return _accel.min_clearance_batch(np.ascontiguousarray(poses), self._data_c, self.grid.origin[0],
                                              self.grid.origin[1], self._res, self._cx, self._cy, a, b, cap)
        rc = math.hypot(a, b)
        for i0 in range(0, len(poses), 256):
            p = poses[i0:i0 + 256]
            c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
            cx = p[:, 0] + c * self._cx - s * self._cy
            cy = p[:, 1] + s * self._cx + c * self._cy
            lists = self._tree.query_ball_point(np.column_stack([cx, cy]), rc + cap + h * 2)
            for j, lst in enumerate(lists):
                if lst:
                    d = rect_square_distance(cx[j], cy[j], c[j], s[j], a, b, self._bcells[lst], h)
                    out[i0 + j] = min(cap, float(d.min()))
            # only boundary cells are indexed: a footprint inside a thick obstacle
            # (spine point in an occupied cell) has zero clearance
            swx, swy = self._local_points(p, self._spine_x, self._cy)
            out[i0:i0 + len(p)][np.any(self.grid.is_occupied(swx, swy), axis=1)] = 0.0
        return out


def rect_square_distance(pcx, pcy, c, s, a, b, q: np.ndarray, h: float) -> np.ndarray:
    """Exact distance between one oriented rectangle and many axis-aligned
    squares (centres ``q`` (N,2), half size ``h``); 0 where they intersect.
    For disjoint convex polygons the minimum is attained between a vertex of
    one polygon and the other polygon, so vertex-to-polygon distances suffice."""
    dx = q[:, 0] - pcx
    dy = q[:, 1] - pcy
    ac, as_ = abs(c), abs(s)
    overlap = ~((np.abs(dx) > h + a * ac + b * as_) | (np.abs(dy) > h + a * as_ + b * ac)
                | (np.abs(dx * c + dy * s) > a + h * (ac + as_)) | (np.abs(-dx * s + dy * c) > b + h * (ac + as_)))
    best = np.full(len(q), np.inf)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            px, py = dx + sx * h, dy + sy * h                    # square corner -> rectangle
            lx = np.abs(px * c + py * s) - a
            ly = np.abs(-px * s + py * c) - b
            best = np.minimum(best, np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)))
            rx, ry = sx * a * c - sy * b * s, sx * a * s + sy * b * c  # rectangle corner -> square
            ex = np.abs(rx - dx) - h
            ey = np.abs(ry - dy) - h
            best = np.minimum(best, np.hypot(np.maximum(ex, 0), np.maximum(ey, 0)))
    return np.where(overlap, 0.0, best)


class ZonedChecker:
    """Collision checker with a reduced margin against the obstacles that make
    the start / goal tight.

    Real vehicles often start (or must park) closer to an obstacle than the
    nominal safety margin. For such an endpoint a zone is created with

    * ``radius``   : only poses whose rear axle is within this distance of the
      endpoint may use the relaxation,
    * ``relaxed``  : checker with the reduced margin (the endpoint's own
      clearance, never below ``min_relaxed_margin``) on the full map,
    * ``others``   : checker with the NOMINAL margin on the map *without* the
      cells that constrain the endpoint.

    A pose is accepted if the nominal checker accepts it, or if it lies in a
    zone and both ``relaxed`` and ``others`` accept it: the margin is reduced
    only with respect to the obstacles that were already close to the endpoint;
    every other obstacle keeps the full margin.
    """

    def __init__(self, nominal: CollisionChecker, zones):
        self.nominal = nominal
        self.zones = [(float(x), float(y), float(r), relaxed, others) for (x, y, r, relaxed, others) in zones]
        self.edt = nominal.edt
        self.grid = nominal.grid
        self.vehicle = nominal.vehicle
        self.method = nominal.method
        self.margin = min([nominal.margin] + [z[3].margin for z in self.zones])

    @property
    def check_count(self) -> int:
        return self.nominal.check_count + sum(z[3].check_count + z[4].check_count for z in self.zones)

    @property
    def exact_count(self) -> int:
        return self.nominal.exact_count + sum(z[3].exact_count + z[4].exact_count for z in self.zones)

    def check_poses(self, poses: np.ndarray) -> np.ndarray:
        poses = np.asarray(poses, dtype=float).reshape(-1, 3)
        coll = self.nominal.check_poses(poses)
        for x, y, r, relaxed, others in self.zones:
            if not coll.any():
                break
            idx = np.nonzero(coll & (np.hypot(poses[:, 0] - x, poses[:, 1] - y) <= r))[0]
            if len(idx):
                ok = ~relaxed.check_poses(poses[idx])
                if ok.any():
                    ok[ok] = ~others.check_poses(poses[idx[ok]])
                coll[idx[ok]] = False
        return coll

    def check_pose(self, x: float, y: float, yaw: float) -> bool:
        return bool(self.check_poses(np.array([[x, y, yaw]]))[0])

    def any_collision(self, poses: np.ndarray) -> bool:
        if not self.zones:
            return self.nominal.any_collision(poses)
        return bool(self.check_poses(poses).any())

    def clearance_estimate(self, poses: np.ndarray) -> np.ndarray:
        return self.nominal.clearance_estimate(poses)

    def min_clearance(self, poses: np.ndarray, cap: float = 5.0) -> np.ndarray:
        return self.nominal.min_clearance(poses, cap)
