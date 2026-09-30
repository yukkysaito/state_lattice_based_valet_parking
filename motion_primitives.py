"""Automatic motion-primitive generation from vehicle parameters.

Primitives are generated in the local frame of the start pose (rear axle
center, heading along +x). Every primitive stores its relative end pose, the
dense sampled trajectory (used by collision checking), direction, steering and
path length.

Lattice consistency
-------------------
For constant-steer arcs the arc length is chosen so that the heading change is
an integer multiple of ``yaw_resolution``. Starting from a lattice heading the
successor heading therefore lands exactly on the heading lattice, i.e. the
heading dimension is a true state lattice. Positions are kept continuous inside
their cell (hybrid lattice) which avoids snapping error accumulating along the
path while the discretised cell is still used for duplicate detection.

Extensibility
-------------
:class:`PrimitiveGenerator` is the extension point. ``ConstantSteerPrimitiveGenerator``
is the reference implementation; a clothoid or optimization-based generator
only needs to return a list of :class:`MotionPrimitive` objects with sampled
local trajectories.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

from vehicle import VehicleInfo, arc_samples

FORWARD = 1
REVERSE = -1

# Target length of each class expressed as a fraction of the minimum turning
# radius, so that the primitive set scales with the vehicle.
LENGTH_CLASS_FACTORS = {"micro": 0.05, "short": 0.10, "medium": 0.30, "long": 0.75}
# "micro" primitives are NOT heading-lattice consistent (their heading change is
# smaller than one yaw bin); they exist for fine positioning in tight spaces
# (tight parallel parking) and only use straight and full-lock steering.
NON_LATTICE_CLASSES = ("micro",)


@dataclass(frozen=True)
class PrimitiveConfig:
    n_steer: int = 5
    length_classes: Tuple[str, ...] = ("short", "medium")
    yaw_resolution: float = math.radians(5.0)
    xy_resolution: float = 0.1
    sample_ds: float = 0.05
    allow_reverse: bool = True
    generator: str = "constant_steer"

    def validate(self) -> None:
        if self.n_steer < 1 or (self.n_steer % 2) == 0:
            raise ValueError("n_steer must be an odd number >= 1 (straight must be included)")
        for c in self.length_classes:
            if c not in LENGTH_CLASS_FACTORS:
                raise ValueError(f"unknown length class {c!r}")
        if not (0.005 <= self.sample_ds <= 0.2):
            raise ValueError("sample_ds must be within [0.005, 0.2] m")
        if self.yaw_resolution <= 0 or self.xy_resolution <= 0:
            raise ValueError("resolutions must be positive")


@dataclass
class MotionPrimitive:
    index: int
    direction: int            # +1 forward, -1 reverse
    steering: float           # [rad], constant along the primitive
    length: float             # unsigned travelled distance [m]
    length_class: str
    dx: float                 # relative end pose in start frame
    dy: float
    dyaw: float
    samples: np.ndarray       # (N,3) local poses, excluding start, including end
    curvature: float = 0.0

    @property
    def end(self) -> Tuple[float, float, float]:
        return self.dx, self.dy, self.dyaw


@dataclass
class PrimitiveSet:
    """Primitive list plus flattened arrays for vectorised expansion."""

    primitives: List[MotionPrimitive]
    vehicle: VehicleInfo
    config: PrimitiveConfig
    all_samples: np.ndarray = field(init=False)     # (M,3)
    owner: np.ndarray = field(init=False)           # (M,) primitive index per sample
    starts: np.ndarray = field(init=False)          # (P,) first sample index per primitive
    ends: np.ndarray = field(init=False)            # (P,3) relative end pose
    directions: np.ndarray = field(init=False)
    steerings: np.ndarray = field(init=False)
    lengths: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.all_samples = np.concatenate([p.samples for p in self.primitives], axis=0)
        self.owner = np.concatenate(
            [np.full(len(p.samples), i, dtype=np.int64) for i, p in enumerate(self.primitives)])
        counts = np.array([len(p.samples) for p in self.primitives])
        self.starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.ends = np.array([p.end for p in self.primitives])
        self.directions = np.array([p.direction for p in self.primitives], dtype=np.int64)
        self.steerings = np.array([p.steering for p in self.primitives])
        self.lengths = np.array([p.length for p in self.primitives])

    def __len__(self) -> int:
        return len(self.primitives)

    def __iter__(self):
        return iter(self.primitives)

    def transform_ends(self, x: float, y: float, yaw: float) -> np.ndarray:
        return _transform(self.ends, x, y, yaw)

    def transform_samples(self, x: float, y: float, yaw: float, mask: np.ndarray | None = None):
        """World poses of all samples (optionally only for primitives in ``mask``).

        Returns (poses, owner) where owner maps each pose to its primitive index.
        """
        if mask is None:
            return _transform(self.all_samples, x, y, yaw), self.owner
        sel = mask[self.owner]
        return _transform(self.all_samples[sel], x, y, yaw), self.owner[sel]

    def steering_values(self) -> np.ndarray:
        return np.unique(np.round(self.steerings, 9))


def _transform(local: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.empty_like(local)
    out[:, 0] = x + c * local[:, 0] - s * local[:, 1]
    out[:, 1] = y + s * local[:, 0] + c * local[:, 1]
    out[:, 2] = (yaw + local[:, 2] + math.pi) % (2.0 * math.pi) - math.pi
    return out


# -------------------------------------------------------------- generators
class PrimitiveGenerator:
    """Interface for primitive generators (constant steer, clothoid, optimized...)."""

    name = "base"

    def generate(self, vehicle: VehicleInfo, config: PrimitiveConfig) -> List[MotionPrimitive]:
        raise NotImplementedError


def steering_samples(max_steer: float, n: int) -> np.ndarray:
    if n == 1:
        return np.array([0.0])
    vals = np.linspace(-max_steer, max_steer, n)
    vals[n // 2] = 0.0  # exact zero for the straight primitive
    return vals


def target_lengths(vehicle: VehicleInfo, config: PrimitiveConfig) -> dict:
    r = vehicle.min_turning_radius
    min_len = min_primitive_length(config)
    return {c: (LENGTH_CLASS_FACTORS[c] * r if c in NON_LATTICE_CLASSES else max(min_len, LENGTH_CLASS_FACTORS[c] * r))
            for c in config.length_classes}


def min_primitive_length(config: PrimitiveConfig) -> float:
    # a successor must leave the current xy cell, otherwise it is pruned
    return 3.0 * config.xy_resolution


class ConstantSteerPrimitiveGenerator(PrimitiveGenerator):
    name = "constant_steer"

    def generate(self, vehicle: VehicleInfo, config: PrimitiveConfig) -> List[MotionPrimitive]:
        config.validate()
        steers = steering_samples(vehicle.max_steer_angle, config.n_steer)
        targets = target_lengths(vehicle, config)
        min_len = min_primitive_length(config)
        dirs = [FORWARD, REVERSE] if config.allow_reverse else [FORWARD]
        prims: List[MotionPrimitive] = []
        seen = set()
        for direction in dirs:
            for cls in config.length_classes:
                for steer in steers:
                    kappa = math.tan(steer) / vehicle.wheel_base
                    if cls in NON_LATTICE_CLASSES:
                        if 0.0 < abs(steer) < vehicle.max_steer_angle - 1e-9:
                            continue
                        # >= 0.3 m: shorter steering holds read as jerky steering
                        length = max(targets[cls], 2.0 * config.xy_resolution, 0.3)
                    else:
                        length = self._lattice_length(kappa, targets[cls], min_len, config.yaw_resolution)
                    key = (direction, round(steer, 9), round(length, 6))
                    if key in seen:
                        continue
                    seen.add(key)
                    n = max(2, int(math.ceil(length / config.sample_ds)))
                    s = np.linspace(0.0, length, n + 1)[1:]
                    samples = arc_samples(kappa, direction, s)
                    end = samples[-1]
                    prims.append(MotionPrimitive(
                        index=len(prims), direction=direction, steering=float(steer), length=length,
                        length_class=cls, dx=float(end[0]), dy=float(end[1]), dyaw=float(end[2]),
                        samples=samples, curvature=kappa))
        return prims

    @staticmethod
    def _lattice_length(kappa: float, target: float, min_len: float, yaw_res: float) -> float:
        if abs(kappa) < 1e-9:
            return target
        k = max(1, int(round(abs(kappa) * target / yaw_res)))
        length = k * yaw_res / abs(kappa)
        while length < min_len:
            k += 1
            length = k * yaw_res / abs(kappa)
        return length


GENERATORS = {ConstantSteerPrimitiveGenerator.name: ConstantSteerPrimitiveGenerator}


@lru_cache(maxsize=64)
def generate_primitives(vehicle: VehicleInfo, config: PrimitiveConfig) -> PrimitiveSet:
    """Cached factory. The cache key contains every vehicle parameter, so any
    parameter change regenerates the primitive set automatically."""
    gen = GENERATORS[config.generator]()
    return PrimitiveSet(gen.generate(vehicle, config), vehicle, config)


def describe(pset: PrimitiveSet) -> str:
    lines = [f"{len(pset)} primitives for {pset.vehicle.summary()}"]
    for p in pset:
        lines.append(f"  #{p.index:02d} dir={p.direction:+d} steer={math.degrees(p.steering):+6.1f}deg "
                     f"len={p.length:5.2f} ({p.length_class:6s}) end=({p.dx:+.2f},{p.dy:+.2f},"
                     f"{math.degrees(p.dyaw):+6.1f}deg) n={len(p.samples)}")
    return "\n".join(lines)


def primitives_for(vehicle: VehicleInfo, n_steer: int = 5,
                   length_classes: Sequence[str] = ("short", "medium"), **kw) -> PrimitiveSet:
    return generate_primitives(vehicle, PrimitiveConfig(n_steer=n_steer,
                                                        length_classes=tuple(length_classes), **kw))
