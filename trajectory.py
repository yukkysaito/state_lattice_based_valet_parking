"""Trajectory output data structures (Autoware Trajectory-like)."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw: float
    direction: int      # +1 forward, -1 reverse
    steering: float     # [rad]
    curvature: float    # tan(steering) / wheel_base  [1/m]
    arc_length: float   # cumulative unsigned distance from start [m]


class Trajectory:
    """Ordered list of :class:`TrajectoryPoint`.

    A point's ``direction``/``steering`` describe the motion that *arrives* at
    that point (the first point carries the values of the first segment).
    A direction switch (cusp) is the last point before the direction flips.
    """

    def __init__(self, points: List[TrajectoryPoint]):
        self.points = list(points)

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, i):
        return self.points[i]

    @classmethod
    def from_arrays(cls, poses: np.ndarray, directions, steerings, wheel_base: float) -> "Trajectory":
        poses = np.asarray(poses, dtype=float)
        directions = np.asarray(directions, dtype=int)
        steerings = np.asarray(steerings, dtype=float)
        if len(poses) == 0:
            return cls([])
        ds = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(ds)])
        kappa = np.tan(steerings) / wheel_base
        return cls([TrajectoryPoint(float(p[0]), float(p[1]), float(p[2]), int(d), float(st), float(k), float(a))
                    for p, d, st, k, a in zip(poses, directions, steerings, kappa, s)])

    # ------------------------------------------------------------ arrays
    def poses(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 3))
        return np.array([[p.x, p.y, p.yaw] for p in self.points])

    def directions(self) -> np.ndarray:
        return np.array([p.direction for p in self.points], dtype=int)

    def steerings(self) -> np.ndarray:
        return np.array([p.steering for p in self.points])

    def curvatures(self) -> np.ndarray:
        return np.array([p.curvature for p in self.points])

    # ------------------------------------------------------------ metrics
    @property
    def length(self) -> float:
        return self.points[-1].arc_length if self.points else 0.0

    def directional_lengths(self):
        if len(self.points) < 2:
            return 0.0, 0.0
        p = self.poses()
        ds = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
        d = self.directions()[1:]
        return float(ds[d > 0].sum()), float(ds[d < 0].sum())

    def switch_indices(self) -> List[int]:
        d = self.directions()
        return [i for i in range(1, len(d) - 1) if d[i] != d[i + 1]]

    @property
    def n_direction_changes(self) -> int:
        return len(self.switch_indices())

    def split_by_direction(self) -> List["Trajectory"]:
        """Split at cusps; each part shares its cusp point with the next part
        (Autoware freespace planners publish one direction at a time)."""
        if not self.points:
            return []
        cuts = self.switch_indices()
        parts, start = [], 0
        for c in cuts:
            parts.append(Trajectory(self.points[start:c + 1]))
            start = c
        parts.append(Trajectory(self.points[start:]))
        return parts

    def to_autoware_dicts(self, velocity: float = 1.0) -> List[Dict]:
        """Autoware ``TrajectoryPoint``-like dicts (signed longitudinal velocity)."""
        out = []
        for p in self.points:
            out.append({
                "pose": {"position": {"x": p.x, "y": p.y, "z": 0.0},
                         "orientation": {"z": math.sin(p.yaw / 2), "w": math.cos(p.yaw / 2)}},
                "longitudinal_velocity_mps": velocity * p.direction,
                "front_wheel_angle_rad": p.steering,
                "heading_rate_rps": 0.0,
            })
        if out:
            out[-1]["longitudinal_velocity_mps"] = 0.0
        return out

    def to_dicts(self) -> List[Dict]:
        return [asdict(p) for p in self.points]
