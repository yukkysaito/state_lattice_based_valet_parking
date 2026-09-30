"""Vehicle parameters and kinematic model.

The reference point of every pose in this package is the **rear axle center**
(Autoware ``base_link``). All footprint geometry is derived from Autoware
``VehicleInfo``-equivalent parameters, so changing a parameter automatically
changes the footprint, the minimum turning radius, the motion primitives and
the heuristic tables (they are all keyed on :meth:`VehicleInfo.key`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Tuple

import numpy as np


class InvalidVehicleParameterError(ValueError):
    """Raised when vehicle parameters are physically invalid."""


@dataclass(frozen=True)
class VehicleInfo:
    """Autoware VehicleInfo-equivalent parameters (all SI units)."""

    wheel_base: float = 2.79
    wheel_tread: float = 1.60
    front_overhang: float = 0.90
    rear_overhang: float = 1.00
    left_overhang: float = 0.15
    right_overhang: float = 0.15
    max_steer_angle: float = 0.55
    name: str = field(default="default", compare=False)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        errors = []
        for attr in ("wheel_base", "wheel_tread"):
            v = getattr(self, attr)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0):
                errors.append(f"{attr} must be finite and > 0 (got {v!r})")
        for attr in ("front_overhang", "rear_overhang", "left_overhang", "right_overhang"):
            v = getattr(self, attr)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0.0):
                errors.append(f"{attr} must be finite and >= 0 (got {v!r})")
        s = self.max_steer_angle
        if not (isinstance(s, (int, float)) and math.isfinite(s) and 0.0 < s < math.radians(80.0)):
            errors.append(f"max_steer_angle must be in (0, 80deg) (got {s!r})")
        if errors:
            raise InvalidVehicleParameterError("; ".join(errors))

    # --------------------------------------------------------- derived values
    @property
    def vehicle_length(self) -> float:
        return self.front_overhang + self.wheel_base + self.rear_overhang

    @property
    def vehicle_width(self) -> float:
        return self.wheel_tread + self.left_overhang + self.right_overhang

    @property
    def min_turning_radius(self) -> float:
        """Turning radius of the rear axle center at max steering."""
        return self.wheel_base / math.tan(self.max_steer_angle)

    @property
    def max_curvature(self) -> float:
        return math.tan(self.max_steer_angle) / self.wheel_base

    @property
    def footprint_center_offset(self) -> Tuple[float, float]:
        """Center of the footprint rectangle in the rear-axle frame."""
        cx = (self.wheel_base + self.front_overhang - self.rear_overhang) / 2.0
        cy = (self.left_overhang - self.right_overhang) / 2.0
        return cx, cy

    @property
    def half_length(self) -> float:
        return self.vehicle_length / 2.0

    @property
    def half_width(self) -> float:
        return self.vehicle_width / 2.0

    def footprint_local(self, margin=0.0) -> np.ndarray:
        """Footprint polygon (4x2, CCW) in the rear-axle frame.

        ``margin``: one value for every side, or (longitudinal, lateral)."""
        m_lon, m_lat = split_margin(margin)
        front = self.wheel_base + self.front_overhang + m_lon
        rear = -self.rear_overhang - m_lon
        left = self.wheel_tread / 2.0 + self.left_overhang + m_lat
        right = -(self.wheel_tread / 2.0 + self.right_overhang + m_lat)
        return np.array([[front, right], [front, left], [rear, left], [rear, right]])

    def footprint_world(self, x: float, y: float, yaw: float, margin=0.0) -> np.ndarray:
        local = self.footprint_local(margin)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([x, y])

    def key(self) -> Tuple[float, ...]:
        """Hashable identity used to cache vehicle-dependent tables."""
        return (
            round(self.wheel_base, 6), round(self.wheel_tread, 6),
            round(self.front_overhang, 6), round(self.rear_overhang, 6),
            round(self.left_overhang, 6), round(self.right_overhang, 6),
            round(self.max_steer_angle, 6),
        )

    def with_changes(self, **kwargs) -> "VehicleInfo":
        return replace(self, **kwargs)

    def summary(self) -> str:
        return (f"{self.name}: L={self.vehicle_length:.2f} W={self.vehicle_width:.2f} "
                f"WB={self.wheel_base:.2f} steer={math.degrees(self.max_steer_angle):.1f}deg "
                f"Rmin={self.min_turning_radius:.2f}")


def split_margin(margin) -> Tuple[float, float]:
    """(longitudinal, lateral) margin from a scalar or a pair."""
    if isinstance(margin, (tuple, list, np.ndarray)):
        return float(margin[0]), float(margin[1])
    return float(margin), float(margin)


# --------------------------------------------------------------------- model
def normalize_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    r = (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi
    return np.where(r >= np.pi, r - 2.0 * np.pi, r)


def wrap_angle(a: float) -> float:
    r = (a + math.pi) % (2.0 * math.pi) - math.pi
    return r - 2.0 * math.pi if r >= math.pi else r


class KinematicBicycleModel:
    """Low-speed Ackermann model parameterised by signed arc length.

    dx/ds = cos(yaw) * d,  dy/ds = sin(yaw) * d,  dyaw/ds = d * tan(steer) / L
    where ``s`` is the travelled (unsigned) distance and ``d`` is +1 (forward)
    or -1 (reverse). Constant steering is integrated in closed form (exact
    arc), which removes integration error from primitive endpoints.

    ``max_steer_rate`` is stored so that a steering-rate constrained model
    (steering as an additional state) can be added without changing callers.
    """

    def __init__(self, vehicle: VehicleInfo, max_steer_rate: float | None = None):
        self.vehicle = vehicle
        self.max_steer_rate = max_steer_rate

    def curvature(self, steer: float) -> float:
        return math.tan(steer) / self.vehicle.wheel_base

    def propagate(self, x: float, y: float, yaw: float, steer: float, direction: int,
                  length: float, ds: float = 0.05) -> np.ndarray:
        """Integrate a constant-steer segment. Returns (N,3) poses incl. start."""
        if length <= 0.0:
            return np.array([[x, y, yaw]])
        n = max(1, int(math.ceil(length / ds)))
        s = np.linspace(0.0, length, n + 1)
        local = arc_samples(self.curvature(steer), direction, s)
        c, si = math.cos(yaw), math.sin(yaw)
        out = np.empty_like(local)
        out[:, 0] = x + c * local[:, 0] - si * local[:, 1]
        out[:, 1] = y + si * local[:, 0] + c * local[:, 1]
        out[:, 2] = normalize_angle(yaw + local[:, 2])
        return out

    def step_euler(self, x: float, y: float, yaw: float, steer: float, direction: int,
                   ds: float) -> Tuple[float, float, float]:
        """Single explicit Euler step (used for cross-checking the closed form)."""
        d = 1.0 if direction >= 0 else -1.0
        return (x + d * ds * math.cos(yaw), y + d * ds * math.sin(yaw),
                wrap_angle(yaw + d * ds * self.curvature(steer)))


def arc_samples(kappa: float, direction: int, s: np.ndarray) -> np.ndarray:
    """Exact constant-curvature samples in the local frame of the start pose."""
    d = 1.0 if direction >= 0 else -1.0
    ss = d * np.asarray(s, dtype=float)
    out = np.empty((ss.size, 3))
    if abs(kappa) < 1e-9:
        out[:, 0] = ss
        out[:, 1] = 0.0
        out[:, 2] = 0.0
    else:
        th = kappa * ss
        out[:, 0] = np.sin(th) / kappa
        out[:, 1] = (1.0 - np.cos(th)) / kappa
        out[:, 2] = th
    return out


# ----------------------------------------------------------- vehicle presets
def default_vehicle() -> VehicleInfo:
    return VehicleInfo(name="autoware_default")


def small_car() -> VehicleInfo:
    # wheelbase 2.5, width 1.7
    return VehicleInfo(wheel_base=2.50, wheel_tread=1.45, front_overhang=0.80, rear_overhang=0.60,
                       left_overhang=0.125, right_overhang=0.125, max_steer_angle=0.60,
                       name="small_car")


def standard_sedan() -> VehicleInfo:
    # wheelbase 2.8, width 1.8
    return VehicleInfo(wheel_base=2.80, wheel_tread=1.55, front_overhang=0.95, rear_overhang=1.05,
                       left_overhang=0.125, right_overhang=0.125, max_steer_angle=0.55,
                       name="standard_sedan")


def large_suv() -> VehicleInfo:
    # wheelbase 3.0, width 2.0
    return VehicleInfo(wheel_base=3.00, wheel_tread=1.70, front_overhang=1.00, rear_overhang=1.10,
                       left_overhang=0.15, right_overhang=0.15, max_steer_angle=0.55,
                       name="large_suv")


def long_wheelbase() -> VehicleInfo:
    return VehicleInfo(wheel_base=3.60, wheel_tread=1.65, front_overhang=1.00, rear_overhang=1.20,
                       left_overhang=0.15, right_overhang=0.15, max_steer_angle=0.55,
                       name="long_wheelbase")


def reduced_steer_sedan() -> VehicleInfo:
    return standard_sedan().with_changes(max_steer_angle=0.40, name="sedan_low_steer")


VEHICLE_PRESETS = {
    "default": default_vehicle,
    "small": small_car,
    "sedan": standard_sedan,
    "suv": large_suv,
    "long": long_wheelbase,
    "low_steer": reduced_steer_sedan,
}
