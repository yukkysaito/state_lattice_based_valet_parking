"""Optional numba kernels for the collision checker.

The pure-numpy implementation in ``collision_checker.py`` is the reference;
these kernels implement the same cascade (bounds -> circumscribed circle ->
inscribed spine circles -> covering circles -> exact SAT over grid cells) and
are cross-validated in the unit tests. Set ``AUTO_PARKING_NO_NUMBA=1`` to
disable them.
"""
from __future__ import annotations

import math
import os

import numpy as np

HAVE_NUMBA = False
if not os.environ.get("AUTO_PARKING_NO_NUMBA"):
    try:
        import numba  # noqa: F401
        from numba import njit

        HAVE_NUMBA = True
    except Exception:  # pragma: no cover - numba optional
        HAVE_NUMBA = False

if HAVE_NUMBA:
    _BIG = 1.0e6

    @njit(cache=True, inline="always")
    def _lb(px, py, edt, ox, oy, res, h2, W, H):
        ix = int(math.floor((px - ox) / res))
        iy = int(math.floor((py - oy) / res))
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            return -_BIG
        ccx = ox + (ix + 0.5) * res
        ccy = oy + (iy + 0.5) * res
        return edt[iy, ix] - math.hypot(px - ccx, py - ccy) - h2

    @njit(cache=True, inline="always")
    def _ub(px, py, edt, ox, oy, res, W, H):
        ix = int(math.floor((px - ox) / res))
        iy = int(math.floor((py - oy) / res))
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            return _BIG
        ccx = ox + (ix + 0.5) * res
        ccy = oy + (iy + 0.5) * res
        return edt[iy, ix] + math.hypot(px - ccx, py - ccy)

    @njit(cache=True)
    def check_rect_batch(poses, data, edt, ox, oy, res, cx, cy, a, b, spine_x, spine_r,
                         cover_x, cover_r, rcirc, circles_only, circ_x, circ_r, stop_first=False):
        n = poses.shape[0]
        H, W = data.shape
        h = 0.5 * res
        h2 = h * math.sqrt(2.0)
        xmin, ymin = ox, oy
        xmax, ymax = ox + W * res, oy + H * res
        out = np.zeros(n, dtype=np.bool_)
        n_exact = 0
        for i in range(n):
            x, y, yaw = poses[i, 0], poses[i, 1], poses[i, 2]
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
                out[i] = True  # invalid pose: never report it as free
                if stop_first:
                    break
                continue
            c, s = math.cos(yaw), math.sin(yaw)
            # 1. bounds (all 4 corners inside the map)
            inside = True
            bx0, bx1, by0, by1 = 1e18, -1e18, 1e18, -1e18
            for k in range(4):
                lx = cx + a if k < 2 else cx - a
                ly = cy - b if (k == 0 or k == 3) else cy + b
                wx = x + c * lx - s * ly
                wy = y + s * lx + c * ly
                if wx < xmin or wx >= xmax or wy < ymin or wy >= ymax:
                    inside = False
                bx0 = min(bx0, wx); bx1 = max(bx1, wx)
                by0 = min(by0, wy); by1 = max(by1, wy)
            if not inside:
                out[i] = True
                if stop_first:
                    break
                continue
            if circles_only:
                hit = False
                for k in range(circ_x.shape[0]):
                    px = x + c * circ_x[k] - s * cy
                    py = y + s * circ_x[k] + c * cy
                    if _lb(px, py, edt, ox, oy, res, h2, W, H) < circ_r:
                        hit = True
                        break
                out[i] = hit
                if hit and stop_first:
                    break
                continue
            # 2. circumscribed circle
            pcx = x + c * cx - s * cy
            pcy = y + s * cx + c * cy
            if _lb(pcx, pcy, edt, ox, oy, res, h2, W, H) > rcirc:
                continue
            # 3. inscribed spine circles -> sure collision
            hit = False
            for k in range(spine_x.shape[0]):
                px = x + c * spine_x[k] - s * cy
                py = y + s * spine_x[k] + c * cy
                if _ub(px, py, edt, ox, oy, res, W, H) < spine_r:
                    hit = True
                    break
            if hit:
                out[i] = True
                if stop_first:
                    break
                continue
            # 4. covering circles -> sure free
            free = True
            for k in range(cover_x.shape[0]):
                px = x + c * cover_x[k] - s * cy
                py = y + s * cover_x[k] + c * cy
                if _lb(px, py, edt, ox, oy, res, h2, W, H) <= cover_r:
                    free = False
                    break
            if free:
                continue
            # 5. exact SAT over occupied cells in the bounding box
            n_exact += 1
            ix0 = max(int(math.floor((bx0 - ox) / res)), 0)
            ix1 = min(int(math.floor((bx1 - ox) / res)), W - 1)
            iy0 = max(int(math.floor((by0 - oy) / res)), 0)
            iy1 = min(int(math.floor((by1 - oy) / res)), H - 1)
            ac, as_ = abs(c), abs(s)
            ex_x = h + a * ac + b * as_
            ex_y = h + a * as_ + b * ac
            ex_u = a + h * (ac + as_)
            ex_v = b + h * (ac + as_)
            for iy in range(iy0, iy1 + 1):
                qy = oy + (iy + 0.5) * res
                dy = qy - pcy
                if abs(dy) > ex_y:
                    continue
                for ix in range(ix0, ix1 + 1):
                    if data[iy, ix] == 0:
                        continue
                    qx = ox + (ix + 0.5) * res
                    dx = qx - pcx
                    if abs(dx) > ex_x:
                        continue
                    if abs(dx * c + dy * s) > ex_u:
                        continue
                    if abs(-dx * s + dy * c) > ex_v:
                        continue
                    hit = True
                    break
                if hit:
                    break
            out[i] = hit
            if hit and stop_first:
                break
        return out, n_exact

    @njit(cache=True)
    def clearance_estimate_batch(poses, edt, ox, oy, res, cy, spine_x, b0):
        n = poses.shape[0]
        H, W = edt.shape
        h = 0.5 * res
        out = np.empty(n)
        for i in range(n):
            x, y, yaw = poses[i, 0], poses[i, 1], poses[i, 2]
            c, s = math.cos(yaw), math.sin(yaw)
            m = 1e18
            for k in range(spine_x.shape[0]):
                px = x + c * spine_x[k] - s * cy
                py = y + s * spine_x[k] + c * cy
                ix = int(math.floor((px - ox) / res))
                iy = int(math.floor((py - oy) / res))
                if ix < 0 or iy < 0 or ix >= W or iy >= H:
                    d = 0.0
                else:
                    d = max(edt[iy, ix] - h, 0.0)
                if d < m:
                    m = d
            out[i] = max(m - b0, 0.0)
        return out

    @njit(cache=True, inline="always")
    def rect_square_distance(pcx, pcy, c, s, a, b, qx, qy, h):
        """Exact distance between an oriented rectangle (centre, axes (c,s), half
        extents a, b) and an axis-aligned square (centre q, half size h); 0 if
        they intersect."""
        dx = qx - pcx
        dy = qy - pcy
        ac, as_ = abs(c), abs(s)
        if not (abs(dx) > h + a * ac + b * as_ or abs(dy) > h + a * as_ + b * ac
                or abs(dx * c + dy * s) > a + h * (ac + as_) or abs(-dx * s + dy * c) > b + h * (ac + as_)):
            return 0.0
        best = 1e18
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                # square corner -> rectangle
                px = dx + sx * h
                py = dy + sy * h
                lx = abs(px * c + py * s) - a
                ly = abs(-px * s + py * c) - b
                d = math.hypot(max(lx, 0.0), max(ly, 0.0))
                if d < best:
                    best = d
                # rectangle corner -> square
                rx = sx * a * c - sy * b * s
                ry = sx * a * s + sy * b * c
                ex = abs(rx - dx) - h
                ey = abs(ry - dy) - h
                d = math.hypot(max(ex, 0.0), max(ey, 0.0))
                if d < best:
                    best = d
        return best

    @njit(cache=True)
    def min_clearance_batch(poses, data, ox, oy, res, cx, cy, a, b, cap):
        """Exact distance from each (bare or inflated) footprint to the nearest
        occupied cell square, capped at ``cap``."""
        n = poses.shape[0]
        H, W = data.shape
        h = 0.5 * res
        out = np.empty(n)
        r = math.hypot(a, b) + cap + res
        for i in range(n):
            x, y, yaw = poses[i, 0], poses[i, 1], poses[i, 2]
            c, s = math.cos(yaw), math.sin(yaw)
            pcx = x + c * cx - s * cy
            pcy = y + s * cx + c * cy
            ix0 = max(int(math.floor((pcx - r - ox) / res)), 0)
            ix1 = min(int(math.floor((pcx + r - ox) / res)), W - 1)
            iy0 = max(int(math.floor((pcy - r - oy) / res)), 0)
            iy1 = min(int(math.floor((pcy + r - oy) / res)), H - 1)
            best = cap
            for iy in range(iy0, iy1 + 1):
                qy = oy + (iy + 0.5) * res
                for ix in range(ix0, ix1 + 1):
                    if data[iy, ix] == 0:
                        continue
                    d = rect_square_distance(pcx, pcy, c, s, a, b, ox + (ix + 0.5) * res, qy, h)
                    if d < best:
                        best = d
            out[i] = best
        return out

    @njit(cache=True)
    def rs_table_lookup(table, x, y, phi, step, dyaw):
        """Trilinear lookup of the RS table folded in y only. ``x`` must already be
        shifted so that x=0 corresponds to table index 0 (same as numpy version)."""
        nxt = table.shape[0]
        n = table.shape[1]
        n_yaw = table.shape[2]
        m = x.shape[0]
        out = np.empty(m)
        two_pi = 2.0 * math.pi
        for i in range(m):
            xi, yi, pi_ = x[i], y[i], phi[i]
            if yi < 0:
                yi = -yi
                pi_ = -pi_
            fx = xi / step
            fy = yi / step
            if fx < 0 or fx >= nxt - 1 or fy >= n - 1:
                out[i] = np.nan
                continue
            fp = ((pi_ + math.pi) % two_pi) / dyaw
            ix = int(fx)
            iy = int(fy)
            ipf = math.floor(fp)
            ip = int(ipf) % n_yaw
            ip1 = (ip + 1) % n_yaw
            tx = fx - ix
            ty = fy - iy
            tp = fp - ipf
            c00 = table[ix, iy, ip] * (1 - tp) + table[ix, iy, ip1] * tp
            c10 = table[ix + 1, iy, ip] * (1 - tp) + table[ix + 1, iy, ip1] * tp
            c01 = table[ix, iy + 1, ip] * (1 - tp) + table[ix, iy + 1, ip1] * tp
            c11 = table[ix + 1, iy + 1, ip] * (1 - tp) + table[ix + 1, iy + 1, ip1] * tp
            out[i] = (c00 * (1 - tx) + c10 * tx) * (1 - ty) + (c01 * (1 - tx) + c11 * tx) * ty
        return out
