"""Reeds-Shepp curves (scalar path generation + vectorised length).

Formulas follow Reeds & Shepp (1990) with the corrections used in OMPL's
``ReedsSheppStateSpace`` (formula numbers 8.1 - 8.11). All computations are
done for a unit turning radius and scaled afterwards.

* :func:`rs_paths` returns every admissible candidate path (used for analytic
  expansion, where the cheapest *collision-free* candidate is selected).
  Each candidate is verified by integrating it and comparing the endpoint;
  numerically inconsistent candidates are dropped (defensive programming).
* :func:`rs_length_vec` computes the shortest length for arrays of relative
  poses (used to build the heuristic lookup table).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

PI = math.pi
HALF_PI = 0.5 * math.pi
ZERO = 1e-9


def _mod2pi(x: float) -> float:
    v = math.fmod(x, 2.0 * PI)
    if v < -PI:
        v += 2.0 * PI
    elif v > PI:
        v -= 2.0 * PI
    return v


def _polar(x: float, y: float) -> Tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _tau_omega(u, v, xi, eta, phi):
    delta = _mod2pi(u - v)
    a = math.sin(u) - math.sin(delta)
    b = math.cos(u) - math.cos(delta) - 1.0
    t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
    tau = _mod2pi(t1 + PI) if t2 < 0 else _mod2pi(t1)
    omega = _mod2pi(tau - u + v - phi)
    return tau, omega


# --------------------------------------------------------- scalar base words
def _LpSpLp(x, y, phi):
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= -ZERO:
        v = _mod2pi(phi - t)
        if v >= -ZERO:
            return t, u, v
    return None


def _LpSpRp(x, y, phi):
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 * u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = _mod2pi(t1 + theta)
        v = _mod2pi(t - phi)
        if t >= -ZERO and v >= -ZERO:
            return t, u, v
    return None


def _LpRmL(x, y, phi):
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = _mod2pi(theta + 0.5 * u + PI)
        v = _mod2pi(phi - t + u)
        if t >= -ZERO and u <= ZERO:
            return t, u, v
    return None


def _LpRupLumRm(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.hypot(xi, eta))
    if rho <= 1.0:
        u = math.acos(rho)
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= -ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRumLumRp(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -HALF_PI:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def _LpRmSmLm(x, y, phi):
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = _polar(xi, eta)
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = _mod2pi(theta + math.atan2(r, -2.0))
        v = _mod2pi(phi - HALF_PI - t)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRmSmRm(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = _mod2pi(t + HALF_PI - phi)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _LpRmSLmRp(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= ZERO:
            t = _mod2pi(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = _mod2pi(t - phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


# (base word, backward?, types, segment builder)
_FAMILIES = [
    (_LpSpLp, False, "LSL", lambda t, u, v: (t, u, v)),
    (_LpSpRp, False, "LSR", lambda t, u, v: (t, u, v)),
    (_LpRmL, False, "LRL", lambda t, u, v: (t, u, v)),
    (_LpRmL, True, "LRL", lambda t, u, v: (v, u, t)),
    (_LpRupLumRm, False, "LRLR", lambda t, u, v: (t, u, -u, v)),
    (_LpRumLumRp, False, "LRLR", lambda t, u, v: (t, u, u, v)),
    (_LpRmSmLm, False, "LRSL", lambda t, u, v: (t, -HALF_PI, u, v)),
    (_LpRmSmLm, True, "LSRL", lambda t, u, v: (v, u, -HALF_PI, t)),
    (_LpRmSmRm, False, "LRSR", lambda t, u, v: (t, -HALF_PI, u, v)),
    (_LpRmSmRm, True, "RSRL", lambda t, u, v: (v, u, -HALF_PI, t)),
    (_LpRmSLmRp, False, "LRSLR", lambda t, u, v: (t, -HALF_PI, u, -HALF_PI, v)),
]

_REFLECT = str.maketrans("LR", "RL")


@dataclass
class RSPath:
    types: str                 # e.g. "LSR"
    lengths: Tuple[float, ...]  # signed segment lengths [m]; negative = reverse
    radius: float

    @property
    def total_length(self) -> float:
        return float(sum(abs(l) for l in self.lengths))

    def segments(self):
        """Yield (type, signed_length) skipping zero-length segments."""
        for t, l in zip(self.types, self.lengths):
            if abs(l) > 1e-6:
                yield t, l

    def shortest_segment(self) -> float:
        return min((abs(l) for _, l in self.segments()), default=math.inf)

    def pieces(self, start):
        """[(direction, length, [steer_sign], end_x, end_y), ...] per segment."""
        out = []
        x, y, yaw = 0.0, 0.0, 0.0
        c, s = math.cos(start[2]), math.sin(start[2])
        for t, l in self.segments():
            x, y, yaw = _integrate_unit(t, (l / self.radius,), x, y, yaw)
            wx = start[0] + self.radius * (c * x - s * y)
            wy = start[1] + self.radius * (s * x + c * y)
            out.append((1 if l > 0 else -1, abs(l), [1 if t == "L" else (-1 if t == "R" else 0)], wx, wy))
        return out

    def direction_runs(self):
        """Lengths of the maximal same-direction runs: [(direction, length), ...]."""
        runs = []
        for _, l in self.segments():
            d = 1 if l > 0 else -1
            if runs and runs[-1][0] == d:
                runs[-1] = (d, runs[-1][1] + abs(l))
            else:
                runs.append((d, abs(l)))
        return runs

    def shortest_run(self, prev_dir: int = 0) -> float:
        """Shortest run between two cusps; the first run is ignored when it continues
        the vehicle's previous direction (it extends the preceding motion)."""
        runs = self.direction_runs()
        if runs and prev_dir != 0 and runs[0][0] == prev_dir:
            runs = runs[1:]
        return min((l for _, l in runs), default=math.inf)

    @property
    def n_direction_changes(self) -> int:
        dirs = [1 if l > 0 else -1 for _, l in self.segments()]
        return sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)


def _unit_candidates(x: float, y: float, phi: float) -> List[Tuple[str, Tuple[float, ...]]]:
    out = []
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    for func, backward, types, build in _FAMILIES:
        bx, by = (xb, yb) if backward else (x, y)
        for (vx, vy, vphi, flip, refl) in ((bx, by, phi, False, False), (-bx, by, -phi, True, False),
                                           (bx, -by, -phi, False, True), (-bx, -by, phi, True, True)):
            r = func(vx, vy, vphi)
            if r is None:
                continue
            segs = build(*r)
            if flip:
                segs = tuple(-s for s in segs)
            ty = types.translate(_REFLECT) if refl else types
            out.append((ty, segs))
    return out


def _integrate_unit(types: str, segs, x=0.0, y=0.0, yaw=0.0):
    for t, v in zip(types, segs):
        if t == "L":
            x, y = x + math.sin(yaw + v) - math.sin(yaw), y - math.cos(yaw + v) + math.cos(yaw)
            yaw += v
        elif t == "R":
            x, y = x - math.sin(yaw - v) + math.sin(yaw), y + math.cos(yaw - v) - math.cos(yaw)
            yaw -= v
        else:
            x, y = x + v * math.cos(yaw), y + v * math.sin(yaw)
    return x, y, yaw


def rs_paths(start, goal, radius: float, verify_tol: float = 1e-4) -> List[RSPath]:
    """All candidate Reeds-Shepp paths from ``start`` to ``goal`` (x, y, yaw),
    sorted by length. Every candidate is endpoint-verified."""
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    x = (c * dx + s * dy) / radius
    y = (-s * dx + c * dy) / radius
    phi = _mod2pi(goal[2] - start[2])
    out = []
    for types, segs in _unit_candidates(x, y, phi):
        ex, ey, eyaw = _integrate_unit(types, segs)
        if (abs(ex - x) > verify_tol or abs(ey - y) > verify_tol
                or abs(_mod2pi(eyaw - phi)) > verify_tol):
            continue
        out.append(RSPath(types, tuple(l * radius for l in segs), radius))
    out.sort(key=lambda p: p.total_length)
    return out


def rs_shortest_length(start, goal, radius: float) -> float:
    paths = rs_paths(start, goal, radius)
    return paths[0].total_length if paths else math.inf


def sample_rs_path(path: RSPath, start, ds: float = 0.05):
    """Sample a path (vectorised per segment).

    Returns (poses (N,3) excluding start, steer_sign (N,), direction (N,), segment_id (N,))
    where steer_sign is +1 (left), -1 (right) or 0 (straight).
    """
    x, y, yaw = start
    r = path.radius
    P, S, D, K = [], [], [], []
    for k, (t, l) in enumerate(path.segments()):
        n = max(1, int(math.ceil(abs(l) / ds)))
        v = l * np.arange(1, n + 1) / n
        sg = 1 if t == "L" else (-1 if t == "R" else 0)
        if sg == 0:
            px = x + v * math.cos(yaw)
            py = y + v * math.sin(yaw)
            pyaw = np.full(n, yaw)
        else:
            phi = v / r * sg
            px = x + r * sg * (np.sin(yaw + phi) - math.sin(yaw))
            py = y - r * sg * (np.cos(yaw + phi) - math.cos(yaw))
            pyaw = yaw + phi
        P.append(np.column_stack([px, py, pyaw]))
        S.append(np.full(n, sg))
        D.append(np.full(n, 1 if l > 0 else -1))
        K.append(np.full(n, k))
        x, y, yaw = float(px[-1]), float(py[-1]), float(pyaw[-1])
    if not P:
        z = np.zeros(0, dtype=int)
        return np.zeros((0, 3)), z, z, z
    poses = np.vstack(P)
    poses[:, 2] = (poses[:, 2] + PI) % (2 * PI) - PI
    return poses, np.concatenate(S), np.concatenate(D), np.concatenate(K)


# ------------------------------------------------------- vectorised lengths
def _vmod2pi(x):
    v = np.fmod(x, 2.0 * PI)
    v = np.where(v < -PI, v + 2.0 * PI, v)
    return np.where(v > PI, v - 2.0 * PI, v)


def _v_tau_omega(u, v, xi, eta, phi):
    delta = _vmod2pi(u - v)
    a = np.sin(u) - np.sin(delta)
    b = np.cos(u) - np.cos(delta) - 1.0
    t1 = np.arctan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (np.cos(delta) - np.cos(v) - np.cos(u)) + 3.0
    tau = np.where(t2 < 0, _vmod2pi(t1 + PI), _vmod2pi(t1))
    omega = _vmod2pi(tau - u + v - phi)
    return tau, omega


# Each vectorised base word returns (ok, t, u, v) exactly like the scalar one.
def _v_LpSpLp(x, y, phi):
    xi, eta = x - np.sin(phi), y - 1.0 + np.cos(phi)
    u, t = np.hypot(xi, eta), np.arctan2(eta, xi)
    v = _vmod2pi(phi - t)
    return (t >= -ZERO) & (v >= -ZERO), t, u, v


def _v_LpSpRp(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    u1 = xi * xi + eta * eta
    t1 = np.arctan2(eta, xi)
    u = np.sqrt(np.maximum(u1 - 4.0, 0.0))
    t = _vmod2pi(t1 + np.arctan2(2.0, u))
    v = _vmod2pi(t - phi)
    return (u1 >= 4.0) & (t >= -ZERO) & (v >= -ZERO), t, u, v


def _v_LpRmL(x, y, phi):
    xi, eta = x - np.sin(phi), y - 1.0 + np.cos(phi)
    u1, theta = np.hypot(xi, eta), np.arctan2(eta, xi)
    u = -2.0 * np.arcsin(np.clip(0.25 * u1, -1.0, 1.0))
    t = _vmod2pi(theta + 0.5 * u + PI)
    v = _vmod2pi(phi - t + u)
    return (u1 <= 4.0) & (t >= -ZERO) & (u <= ZERO), t, u, v


def _v_LpRupLumRm(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho = 0.25 * (2.0 + np.hypot(xi, eta))
    u = np.arccos(np.clip(rho, -1.0, 1.0))
    t, v = _v_tau_omega(u, -u, xi, eta, phi)
    return (rho <= 1.0) & (t >= -ZERO) & (v <= ZERO), t, u, v


def _v_LpRumLumRp(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    u = -np.arccos(np.clip(rho, -1.0, 1.0))
    t, v = _v_tau_omega(u, u, xi, eta, phi)
    return (rho >= 0.0) & (rho <= 1.0) & (u >= -HALF_PI) & (t >= -ZERO) & (v >= -ZERO), t, u, v


def _v_LpRmSmLm(x, y, phi):
    xi, eta = x - np.sin(phi), y - 1.0 + np.cos(phi)
    rho, theta = np.hypot(xi, eta), np.arctan2(eta, xi)
    r = np.sqrt(np.maximum(rho * rho - 4.0, 0.0))
    u = 2.0 - r
    t = _vmod2pi(theta + np.arctan2(r, -2.0))
    v = _vmod2pi(phi - HALF_PI - t)
    return (rho >= 2.0) & (t >= -ZERO) & (u <= ZERO) & (v <= ZERO), t, u, v


def _v_LpRmSmRm(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho, theta = np.hypot(-eta, xi), np.arctan2(xi, -eta)
    t = theta
    u = 2.0 - rho
    v = _vmod2pi(t + HALF_PI - phi)
    return (rho >= 2.0) & (t >= -ZERO) & (u <= ZERO) & (v <= ZERO), t, u, v


def _v_LpRmSLmRp(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho = np.hypot(xi, eta)
    u = 4.0 - np.sqrt(np.maximum(rho * rho - 4.0, 0.0))
    t = _vmod2pi(np.arctan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
    v = _vmod2pi(t - phi)
    return (rho >= 2.0) & (u <= ZERO) & (t >= -ZERO) & (v >= -ZERO), t, u, v


_V_WORDS = {_LpSpLp: _v_LpSpLp, _LpSpRp: _v_LpSpRp, _LpRmL: _v_LpRmL, _LpRupLumRm: _v_LpRupLumRm,
            _LpRumLumRp: _v_LpRumLumRp, _LpRmSmLm: _v_LpRmSmLm, _LpRmSmRm: _v_LpRmSmRm,
            _LpRmSLmRp: _v_LpRmSLmRp}


def rs_length_vec(x, y, phi, radius: float = 1.0) -> np.ndarray:
    """Shortest RS length for arrays of goal poses relative to the start frame."""
    x = np.asarray(x, dtype=float) / radius
    y = np.asarray(y, dtype=float) / radius
    phi = _vmod2pi(np.asarray(phi, dtype=float))
    best = np.full(np.broadcast(x, y, phi).shape, np.inf)
    xb = x * np.cos(phi) + y * np.sin(phi)
    yb = x * np.sin(phi) - y * np.cos(phi)
    with np.errstate(invalid="ignore", divide="ignore"):
        for func, backward, _, build in _FAMILIES:
            vfunc = _V_WORDS[func]
            bx, by = (xb, yb) if backward else (x, y)
            for vx, vy, vphi in ((bx, by, phi), (-bx, by, -phi), (bx, -by, -phi), (-bx, -by, phi)):
                ok, t, u, v = vfunc(vx, vy, vphi)
                length = sum(np.abs(s) for s in build(t, u, v))
                best = np.where(ok & (length < best), length, best)
    return best * radius
