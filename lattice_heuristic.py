"""Free-space lattice heuristic look-up table (HLUT).

The classic state-lattice heuristic (Pivtoraiko & Kelly; Likhachev & Ferguson):
the exact cost-to-go *on the lattice itself* in obstacle-free space, computed
once by a backward Dijkstra from the goal using the **same motion primitives
and the same cost function** as the search (length, reverse weight, steering
weight, direction-switch penalty). Unlike Reeds-Shepp words it captures every
maneuver the lattice can execute (e.g. "straight forward | cusp | reverse arc"),
so it has no plateaus around cusps.

The table is expressed in the goal frame on a grid of ``resolution`` metres,
the planner's heading lattice, and the arrival direction (forward / reverse),
because the switch penalty depends on it. It depends only on the vehicle, the
primitive set and the cost weights, so it is cached in memory and on disk.
Snapping primitive end points to the table grid introduces a small error, so
the value is used as a (tight but not strictly admissible) heuristic.
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
from typing import Dict

import numpy as np

import _accel
from motion_primitives import PrimitiveSet

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

if _accel.HAVE_NUMBA:
    from numba import njit

    @njit(cache=True, inline="always")
    def _heap_push(hk, hv, size, key, val):
        if size >= hk.shape[0]:
            return size, False
        i = size
        hk[i] = key
        hv[i] = val
        while i > 0:
            par = (i - 1) >> 1
            if hk[par] <= hk[i]:
                break
            hk[par], hk[i] = hk[i], hk[par]
            hv[par], hv[i] = hv[i], hv[par]
            i = par
        return size + 1, True

    @njit(cache=True)
    def _backward_dijkstra(n, n_yaw, res, yaw_res, pdx, pdy, pdyaw, pdir, pcost, switch_pen, goal_r, n_levels):
        """table[l, d, ix, iy, iyaw]: cost-to-go of a state whose last motion had
        direction d (0: forward, 1: reverse) and which may still change gear l
        times. ``n_levels == 0``: a single level with an unlimited number of gear
        changes. Grid index i <-> x = (i - (n-1)/2) * res.
        Array-based binary heap with lazy deletion, float64 throughout (the heap
        keys are float64, the stale-entry test must compare in the same precision)."""
        limited = n_levels > 0
        L = n_levels if limited else 1
        table = np.full((L, 2, n, n, n_yaw), np.inf, dtype=np.float64)
        half = (n - 1) // 2
        n_p = pdx.shape[0]
        # predecessor offsets per (primitive, successor heading), snapped to the grid
        offx = np.empty((n_p, n_yaw), dtype=np.int64)
        offy = np.empty((n_p, n_yaw), dtype=np.int64)
        prew = np.empty((n_p, n_yaw), dtype=np.int64)
        for p in range(n_p):
            for iw in range(n_yaw):
                iw0 = (iw - pdyaw[p]) % n_yaw
                yaw0 = iw0 * yaw_res
                c, s = math.cos(yaw0), math.sin(yaw0)
                offx[p, iw] = int(round((c * pdx[p] - s * pdy[p]) / res))
                offy[p, iw] = int(round((s * pdx[p] + c * pdy[p]) / res))
                prew[p, iw] = iw0
        cap = 4 * L * n * n * n_yaw + 1024        # 2 x number of states; overflow is reported
        hk = np.empty(cap, dtype=np.float64)
        hv = np.empty(cap, dtype=np.int64)
        size = 0
        overflow = 0
        stride_l = 2 * n * n * n_yaw
        stride_d = n * n * n_yaw
        stride_x = n * n_yaw
        rg = int(math.ceil(goal_r / res))
        for ix in range(half - rg, half + rg + 1):
            for iy in range(half - rg, half + rg + 1):
                if ((ix - half) * res) ** 2 + ((iy - half) * res) ** 2 <= goal_r * goal_r + 1e-9:
                    for lv in range(L):
                        for d in range(2):
                            table[lv, d, ix, iy, 0] = 0.0
                            size, ok = _heap_push(hk, hv, size, 0.0,
                                                  lv * stride_l + d * stride_d + ix * stride_x + iy * n_yaw)
                            if not ok:
                                overflow += 1
        while size > 0:
            h = hk[0]
            code = hv[0]
            size -= 1
            hk[0] = hk[size]
            hv[0] = hv[size]
            i = 0
            while True:
                l = 2 * i + 1
                if l >= size:
                    break
                r = l + 1
                m = r if (r < size and hk[r] < hk[l]) else l
                if hk[i] <= hk[m]:
                    break
                hk[i], hk[m] = hk[m], hk[i]
                hv[i], hv[m] = hv[m], hv[i]
                i = m
            lv = code // stride_l
            rem = code - lv * stride_l
            d = rem // stride_d
            rem -= d * stride_d
            ix = rem // stride_x
            rem -= ix * stride_x
            iy = rem // n_yaw
            iw = rem - iy * n_yaw
            if h > table[lv, d, ix, iy, iw]:
                continue
            for p in range(n_p):
                dp = 0 if pdir[p] > 0 else 1
                if dp != d:
                    continue
                jx = ix - offx[p, iw]
                jy = iy - offy[p, iw]
                if jx < 0 or jy < 0 or jx >= n or jy >= n:
                    continue
                iw0 = prew[p, iw]
                for dprev in range(2):
                    switch = dprev != dp
                    # a gear change consumes one of the predecessor's remaining changes
                    lp = lv + 1 if (switch and limited) else lv
                    if lp >= L:
                        continue
                    v = h + pcost[p] + (switch_pen if switch else 0.0)
                    if v < table[lp, dprev, jx, jy, iw0]:
                        table[lp, dprev, jx, jy, iw0] = v
                        size, ok = _heap_push(hk, hv, size, v,
                                              lp * stride_l + dprev * stride_d + jx * stride_x + jy * n_yaw + iw0)
                        if not ok:
                            overflow += 1
        return table, overflow

    @njit(cache=True)
    def _lookup(table, res, yaw_res, x, y, phi, d, lv):
        """Bilinear (xy) / nearest (yaw) lookup of the cost from pose (x, y, yaw)
        given in the goal frame, arrival direction d (-1 = unknown) and remaining
        gear changes lv. NaN outside the table or where the table has no finite value."""
        L = table.shape[0]
        n = table.shape[2]
        n_yaw = table.shape[4]
        half = (n - 1) // 2
        m = x.shape[0]
        out = np.empty(m)
        for i in range(m):
            fx = x[i] / res + half
            fy = y[i] / res + half
            if fx < 0 or fy < 0 or fx >= n - 1 or fy >= n - 1:
                out[i] = np.nan
                continue
            li = min(max(lv[i], 0), L - 1)
            iw = int(round(phi[i] / yaw_res)) % n_yaw
            ix = int(fx)
            iy = int(fy)
            tx = fx - ix
            ty = fy - iy
            best = np.inf
            for dd in range(2):
                if d[i] >= 0 and dd != d[i]:
                    continue
                v00 = table[li, dd, ix, iy, iw]
                v10 = table[li, dd, ix + 1, iy, iw]
                v01 = table[li, dd, ix, iy + 1, iw]
                v11 = table[li, dd, ix + 1, iy + 1, iw]
                if not (np.isfinite(v00) and np.isfinite(v10) and np.isfinite(v01) and np.isfinite(v11)):
                    acc = min(min(v00, v10), min(v01, v11))
                else:
                    acc = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty
                if acc < best:
                    best = acc
            # inf (unreachable on the snapped free-space lattice with lv changes) is
            # NOT a proof for continuous poses: report NaN -> sound fallback heuristic
            out[i] = best if np.isfinite(best) else np.nan
        return out


class LatticeHLUT:
    VERSION = 5
    _mem: Dict[str, "LatticeHLUT"] = {}

    def __init__(self, table: np.ndarray, resolution: float, yaw_resolution: float):
        self.table = table
        self.res = resolution
        self.yaw_res = yaw_resolution
        self.extent = (table.shape[2] - 1) / 2 * resolution
        self.levels = table.shape[0]

    @classmethod
    def available(cls) -> bool:
        return _accel.HAVE_NUMBA

    @classmethod
    def get(cls, pset: PrimitiveSet, reverse_weight: float, steer_weight: float, switch_penalty: float,
            extent: float = 16.0, resolution: float = 0.25, goal_radius: float = 0.15,
            forward_weight: float = 1.0, switch_levels: int = 0) -> "LatticeHLUT":
        v = pset.vehicle
        cfg = pset.config
        max_steer = v.max_steer_angle
        cost = pset.lengths * np.where(pset.directions > 0, forward_weight, reverse_weight)
        cost = cost + steer_weight * pset.lengths * np.abs(pset.steerings) / max_steer
        dyaw_steps = np.round(pset.ends[:, 2] / cfg.yaw_resolution).astype(np.int64)
        key_src = repr((cls.VERSION, v.key(), tuple(np.round(pset.ends.ravel(), 6)), tuple(np.round(cost, 6)),
                        tuple(pset.directions), round(cfg.yaw_resolution, 9), switch_penalty, extent,
                        resolution, goal_radius, switch_levels))
        key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
        if key in cls._mem:
            return cls._mem[key]
        n_yaw = int(round(2 * math.pi / cfg.yaw_resolution))
        n = 2 * int(round(extent / resolution)) + 1
        L = switch_levels if switch_levels > 0 else 1
        path = os.path.join(_CACHE_DIR, f"hlut_v{cls.VERSION}_{key}.npy")
        table = None
        if os.path.exists(path):
            try:
                table = np.load(path)
                if table.shape != (L, 2, n, n, n_yaw):
                    table = None
            except Exception:
                table = None
        if table is None:
            table, overflow = _backward_dijkstra(n, n_yaw, resolution, cfg.yaw_resolution,
                                       pset.ends[:, 0].copy(), pset.ends[:, 1].copy(), dyaw_steps,
                                       pset.directions.astype(np.int64), cost.astype(np.float64),
                                       float(switch_penalty), float(max(goal_radius, resolution * 0.5)),
                                       int(switch_levels))
            if overflow:
                raise RuntimeError(f"lattice heuristic Dijkstra heap overflow ({overflow} pushes dropped)")
            # computed in float64 (exact Dijkstra); stored / looked up in float32
            table = table.astype(np.float32)
            try:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".npy")
                with os.fdopen(fd, "wb") as f:
                    np.save(f, table)
                os.replace(tmp, path)
            except OSError:
                pass
        obj = cls(table, resolution, cfg.yaw_resolution)
        cls._mem[key] = obj
        return obj

    def lookup(self, x, y, phi, direction, remaining=None) -> np.ndarray:
        """Pose (x, y, phi) in the goal frame; direction +1/-1/0 (0 = unknown);
        remaining: gear changes still allowed (ignored for an unlimited table)."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        d = np.where(np.asarray(direction) > 0, 0, np.where(np.asarray(direction) < 0, 1, -1)).astype(np.int64)
        if remaining is None or self.levels == 1:
            lv = np.zeros(x.shape, dtype=np.int64)
        else:
            lv = np.ascontiguousarray(np.broadcast_to(np.asarray(remaining, dtype=np.int64), x.shape))
        out = _lookup(self.table, self.res, self.yaw_res, x, np.ascontiguousarray(y, dtype=np.float64),
                      np.ascontiguousarray(phi, dtype=np.float64), np.ascontiguousarray(np.broadcast_to(d, x.shape)),
                      lv)
        return out


class MultiResolutionHLUT:
    """Fine table near the goal, coarser tables further out (first hit wins)."""

    def __init__(self, tables):
        self.tables = list(tables)

    def lookup(self, x, y, phi, direction, remaining=None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = self.tables[0].lookup(x, y, phi, direction, remaining)
        for t in self.tables[1:]:
            miss = np.isnan(out)
            if not miss.any():
                break
            d = np.broadcast_to(np.asarray(direction), x.shape)
            rem = None if remaining is None else np.broadcast_to(np.asarray(remaining), x.shape)[miss]
            out[miss] = t.lookup(x[miss], np.asarray(y, dtype=float)[miss], np.asarray(phi, dtype=float)[miss],
                                 d[miss], rem)
        return out
