"""2D occupancy grid map (``data[y, x]``, 0 = free, 1 = occupied).

The planner only depends on this class, never on raw sensor data. Future
inputs (LaserScan, detected object polygons, curbs, walls, parked vehicles)
are fused into the grid through the ``add_*`` / ``mark_points`` helpers.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np


class OccupancyGrid:
    def __init__(self, data: np.ndarray, resolution: float, origin: Tuple[float, float] = (0.0, 0.0)):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("occupancy grid must be 2D [y, x]")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.data = (data > 0).astype(np.uint8)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))

    # ------------------------------------------------------------ factories
    @classmethod
    def empty(cls, width_m: float, height_m: float, resolution: float = 0.1,
              origin: Tuple[float, float] = (0.0, 0.0)) -> "OccupancyGrid":
        w = int(round(width_m / resolution))
        h = int(round(height_m / resolution))
        return cls(np.zeros((h, w), dtype=np.uint8), resolution, origin)

    def copy(self) -> "OccupancyGrid":
        return OccupancyGrid(self.data.copy(), self.resolution, self.origin)

    # ---------------------------------------------------------- geometry
    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) in world coordinates."""
        x0, y0 = self.origin
        return x0, x0 + self.width * self.resolution, y0, y0 + self.height * self.resolution

    def world_to_index(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(np.int64)
        return ix, iy

    def index_to_world(self, ix, iy):
        """Cell center of index (ix, iy)."""
        x = self.origin[0] + (np.asarray(ix) + 0.5) * self.resolution
        y = self.origin[1] + (np.asarray(iy) + 0.5) * self.resolution
        return x, y

    def in_bounds_index(self, ix, iy):
        return (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)

    def in_bounds(self, x, y):
        xmin, xmax, ymin, ymax = self.extent
        x = np.asarray(x)
        y = np.asarray(y)
        return (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)

    def is_occupied(self, x, y):
        """Occupancy at world points; outside the map counts as occupied."""
        ix, iy = self.world_to_index(x, y)
        inside = self.in_bounds_index(ix, iy)
        out = np.ones(np.shape(ix), dtype=bool)
        out[inside] = self.data[iy[inside], ix[inside]] > 0
        return out

    # ------------------------------------------------------------ editing
    def _cell_centers(self):
        xs = self.origin[0] + (np.arange(self.width) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(self.height) + 0.5) * self.resolution
        return xs, ys

    def add_polygon(self, polygon: np.ndarray, value: int = 1) -> None:
        """Mark every cell whose square intersects the (convex or concave) polygon.

        The cell square is used (not only its center) so that a thin obstacle
        is never lost by rasterisation.
        """
        poly = np.asarray(polygon, dtype=float)
        from matplotlib.path import Path

        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        ix0, iy0 = self.world_to_index(xmin, ymin)
        ix1, iy1 = self.world_to_index(xmax, ymax)
        ix0, iy0 = max(int(ix0), 0), max(int(iy0), 0)
        ix1, iy1 = min(int(ix1), self.width - 1), min(int(iy1), self.height - 1)
        if ix0 > ix1 or iy0 > iy1:
            return
        path = Path(poly)
        r = self.resolution
        # supersample each cell (3x3 + corners) to catch partial overlap
        # interior offsets: an edge lying exactly on a cell boundary (float noise)
        # must not mark the neighbouring cell; overlap > 1% of a cell is marked
        offs = np.array([-0.49, -0.25, 0.0, 0.25, 0.49]) * r
        cx, cy = np.meshgrid(self.origin[0] + (np.arange(ix0, ix1 + 1) + 0.5) * r,
                             self.origin[1] + (np.arange(iy0, iy1 + 1) + 0.5) * r)
        hit = np.zeros(cx.shape, dtype=bool)
        for ox in offs:
            for oy in offs:
                pts = np.column_stack([(cx + ox).ravel(), (cy + oy).ravel()])
                hit |= path.contains_points(pts).reshape(cx.shape)
        # cells containing a polygon vertex
        vx, vy = self.world_to_index(poly[:, 0], poly[:, 1])
        sub = self.data[iy0:iy1 + 1, ix0:ix1 + 1]
        sub[hit] = value
        ok = self.in_bounds_index(vx, vy)
        self.data[vy[ok], vx[ok]] = value

    def add_rectangle(self, cx: float, cy: float, length: float, width: float, yaw: float = 0.0,
                      value: int = 1) -> np.ndarray:
        poly = rectangle_polygon(cx, cy, length, width, yaw)
        self.add_polygon(poly, value)
        return poly

    def add_box(self, xmin: float, ymin: float, xmax: float, ymax: float, value: int = 1) -> np.ndarray:
        poly = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        self.add_polygon(poly, value)
        return poly

    def add_circle(self, cx: float, cy: float, radius: float, value: int = 1) -> None:
        xs, ys = self._cell_centers()
        gx, gy = np.meshgrid(xs, ys)
        half = self.resolution / 2.0
        # distance from circle center to the cell square
        dx = np.maximum(np.abs(gx - cx) - half, 0.0)
        dy = np.maximum(np.abs(gy - cy) - half, 0.0)
        self.data[(dx * dx + dy * dy) <= radius * radius] = value

    def add_border(self, thickness: float = 0.2) -> None:
        n = max(1, int(math.ceil(thickness / self.resolution)))
        self.data[:n, :] = 1
        self.data[-n:, :] = 1
        self.data[:, :n] = 1
        self.data[:, -n:] = 1

    def mark_points(self, points: Iterable[Sequence[float]], inflate: float = 0.0) -> None:
        """Fuse point obstacles (e.g. LaserScan hits already transformed to map frame)."""
        pts = np.asarray(list(points), dtype=float).reshape(-1, 2)
        if inflate > 0:
            for p in pts:
                self.add_circle(p[0], p[1], inflate)
            return
        ix, iy = self.world_to_index(pts[:, 0], pts[:, 1])
        ok = self.in_bounds_index(ix, iy)
        self.data[iy[ok], ix[ok]] = 1

    def crop(self, xmin: float, ymin: float, xmax: float, ymax: float) -> "OccupancyGrid":
        """Sub-grid covering [xmin, xmax] x [ymin, ymax] (clipped to the map),
        aligned to this grid's cells; world coordinates are preserved."""
        ix0, iy0 = self.world_to_index(xmin, ymin)
        ix1, iy1 = self.world_to_index(xmax, ymax)
        ix0, iy0 = max(int(ix0), 0), max(int(iy0), 0)
        ix1, iy1 = min(int(ix1), self.width - 1), min(int(iy1), self.height - 1)
        if ix0 == 0 and iy0 == 0 and ix1 == self.width - 1 and iy1 == self.height - 1:
            return self
        origin = (self.origin[0] + ix0 * self.resolution, self.origin[1] + iy0 * self.resolution)
        return OccupancyGrid(self.data[iy0:iy1 + 1, ix0:ix1 + 1].copy(), self.resolution, origin)

    def occupied_centers(self) -> np.ndarray:
        iy, ix = np.nonzero(self.data)
        x, y = self.index_to_world(ix, iy)
        return np.column_stack([x, y])

    def resampled(self, resolution: float) -> "OccupancyGrid":
        """Conservative resampling (any occupied sub-cell -> occupied)."""
        if abs(resolution - self.resolution) < 1e-12:
            return self.copy()
        xmin, xmax, ymin, ymax = self.extent
        out = OccupancyGrid.empty(xmax - xmin, ymax - ymin, resolution, self.origin)
        iy, ix = np.nonzero(self.data)
        if len(ix):
            r = self.resolution
            for ox in (0.001, 0.999):
                for oy in (0.001, 0.999):
                    x = self.origin[0] + (ix + ox) * r
                    y = self.origin[1] + (iy + oy) * r
                    jx, jy = out.world_to_index(x, y)
                    ok = out.in_bounds_index(jx, jy)
                    out.data[jy[ok], jx[ok]] = 1
        return out


def rectangle_polygon(cx: float, cy: float, length: float, width: float, yaw: float = 0.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])
