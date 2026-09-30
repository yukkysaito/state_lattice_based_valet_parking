"""State-lattice A* planner for automated parking.

Search space : (x, y, yaw) discretised by ``xy_resolution`` / ``yaw_resolution``
               (+ arrival direction, so that arriving forward / reverse at the
               same cell are distinct states - required for the switch cost).
Successors   : motion primitives auto-generated from the vehicle parameters.
Collision    : every sample of a primitive is checked with the vehicle
               footprint. The check margin is ``safety_margin`` plus a bound on
               the footprint motion between two samples, so ``safety_margin``
               holds for the *continuous* motion, not only at the samples.
Heuristic    : max(free-space lattice cost-to-go table, obstacle-aware 2D
               Dijkstra) by default.
Goal         : pose with position / yaw tolerance. A Reeds-Shepp *analytic
               expansion* connects the search exactly to the goal pose; the
               candidate is pushed into the open list (not accepted greedily),
               so it competes with lattice paths on total cost.
Safety       : start/goal collision -> immediate failure; unreachability proven
               by the 2D heuristic -> immediate failure; ``max_expanded_nodes``
               and ``max_planning_time`` always bound the search; every
               exception is caught and reported as ``PlanStatus.ERROR``; the
               final trajectory is re-validated before being returned.

Structure
---------
``StateLatticePlanner`` owns everything that depends only on the vehicle and
the configuration (primitives, heuristic tables). ``LatticeSearch`` is one
search on one map; it can be advanced incrementally with ``step()``, which lets
several searches (e.g. forward and goal->start in ``parking_planner``) share a
time budget deterministically.
"""
from __future__ import annotations

import heapq
import math
import time
import traceback
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from collision_checker import CollisionChecker, ZonedChecker
from heuristic import HEURISTIC_MODES, ReedsSheppHeuristic, build_heuristic
from maneuver import ManeuverRules
from motion_primitives import PrimitiveConfig, PrimitiveSet, generate_primitives
from occupancy_grid import OccupancyGrid
from reeds_shepp import rs_paths, sample_rs_path
from trajectory import Trajectory
from vehicle import VehicleInfo, wrap_angle


class PlanStatus(str, Enum):
    SUCCESS = "success"
    START_IN_COLLISION = "start_in_collision"
    GOAL_IN_COLLISION = "goal_in_collision"
    UNREACHABLE = "unreachable"          # proven by the 2D obstacle heuristic
    NO_PATH = "no_path"                  # open list exhausted
    TIMEOUT = "timeout"
    MAX_EXPANSIONS = "max_expansions"
    INVALID_INPUT = "invalid_input"
    ERROR = "error"


#: statuses that are proofs about the problem (retrying with another search is pointless)
PROOF_STATUSES = frozenset({PlanStatus.START_IN_COLLISION, PlanStatus.GOAL_IN_COLLISION,
                            PlanStatus.UNREACHABLE, PlanStatus.INVALID_INPUT})


@dataclass(frozen=True)
class PlannerConfig:
    # --- lattice discretisation
    xy_resolution: float = 0.1
    yaw_resolution: float = math.radians(5.0)
    include_direction_in_state: bool = True
    # --- primitives
    n_steer: int = 5
    length_classes: Tuple[str, ...] = ("short", "medium", "long")
    # multi-resolution expansion: where the vehicle clearance is >= this, only
    # medium/long primitives are used (coarse, fast, natural long moves); near
    # obstacles the short/fine ones (None = always use every primitive)
    open_space_clearance: Optional[float] = 1.5
    sample_ds: float = 0.025              # also bounds the inter-sample sweep margin
    allow_reverse: bool = True
    primitive_generator: str = "constant_steer"
    # hook for steering-rate constraints (None = unconstrained) [rad / primitive]
    max_steer_change: Optional[float] = None
    # --- collision
    collision_method: str = "rectangle"
    safety_margin: float = 0.1             # guaranteed for the continuous motion (lateral / all sides)
    # front/rear margin if different from the lateral one (e.g. smaller at creep
    # speed: fewer manoeuvres in tight parallel slots); None = safety_margin
    safety_margin_longitudinal: Optional[float] = None
    # near the start / goal the margin may be relaxed down to that endpoint's own
    # clearance (vehicle already parked close to a pillar, tight target slot)
    endpoint_relaxation: bool = True
    endpoint_relaxation_radius: Optional[float] = None   # default: vehicle length
    min_relaxed_margin: float = 0.02       # guaranteed clearance never below this when relaxing
    n_circles: int = 4
    # --- cost (lengths in metres, penalties in "metre-equivalent")
    forward_weight: float = 1.0            # multiplier on forward distance (>1 used by exit planning)
    reverse_weight: float = 1.5            # multiplier on reverse distance
    direction_switch_penalty: float = 10.0
    steer_weight: float = 0.3              # * length * |steer|/max_steer
    steer_change_weight: float = 0.5       # * |delta steer|/max_steer
    clearance_weight: float = 1.0          # * length * max(0, 1 - clearance/ref)
    clearance_ref: float = 0.5
    # --- heuristic
    heuristic: str = "lattice_obstacle"    # falls back to rs_obstacle without numba
    hlut_extent: float = 12.0              # free-space lattice table half-size [m]
    hlut_resolution: float = 0.25
    hlut_far_extent: float = 32.0          # coarse second table for far starts
    hlut_far_resolution: float = 0.5
    heuristic_weight: float = 2.0
    obstacle_heuristic_resolution: float = 0.1  # finer = stronger unreachability proofs
    # --- goal
    position_tolerance: float = 0.10
    yaw_tolerance: float = math.radians(2.0)
    # --- analytic expansion (Reeds-Shepp shot to the exact goal)
    use_analytic_expansion: bool = True
    analytic_near_distance: float = 15.0   # always try when RS distance < this
    analytic_interval: int = 10            # otherwise every N expansions
    analytic_max_candidates: int = 6
    analytic_max_backoff: int = 8          # max stride between shots after repeated failures
    # --- manoeuvre structure (gear changes). These are part of the search
    # state, so they are hard constraints, not just costs:
    # (see maneuver.py)
    max_gear_switches: Optional[int] = 3          # at most this many direction changes (None = unlimited)
    min_maneuver_length: float = 1.0              # minimum manoeuvre length in the aisle / on the road
    min_slot_maneuver_length: float = 0.4         # ... within slot_zone_radius of the slot (in-slot corrections)
    slot_zone_radius: Optional[float] = None      # default: half the vehicle length (car mostly in the slot)
    preferred_maneuver_length: float = 2.0        # soft: a gear change after a shorter manoeuvre is penalised
    short_maneuver_weight: float = 10.0           # penalty = weight * (1 - run / preferred)
    steer_reversal_penalty: float = 5.0           # per left <-> right steering reversal within a manoeuvre
    min_rs_segment_length: float = 0.3            # no Reeds-Shepp arc/straight shorter than this (steering spikes)
    # --- region of interest: the map is cropped to the start/goal bounding box
    # grown by this margin, which bounds the set-up cost (distance field, 2D
    # Dijkstra) on large maps. Outside the ROI counts as occupied (conservative).
    roi_margin: Optional[float] = 25.0
    # --- limits
    max_expanded_nodes: int = 200_000     # deterministic budget (bidirectional: all searches together)
    max_planning_time: float = 20.0
    record_explored: bool = True
    final_validation: bool = True

    def validate(self) -> None:
        if self.xy_resolution <= 0 or self.yaw_resolution <= 0:
            raise ValueError("resolutions must be positive")
        n_yaw = 2 * math.pi / self.yaw_resolution
        if abs(n_yaw - round(n_yaw)) > 1e-6:
            raise ValueError("yaw_resolution must divide 2*pi (e.g. 2.5, 5, 10 deg)")
        if self.collision_method not in CollisionChecker.METHODS:
            raise ValueError(f"unknown collision method {self.collision_method!r}")
        if self.heuristic not in HEURISTIC_MODES:
            raise ValueError(f"unknown heuristic {self.heuristic!r}")
        if self.heuristic_weight < 1.0:
            raise ValueError("heuristic_weight must be >= 1")
        if self.reverse_weight < 1.0 or self.forward_weight < 1.0:
            raise ValueError("forward/reverse weights must be >= 1 (keeps the heuristic admissible)")
        if self.safety_margin_longitudinal is not None and self.safety_margin_longitudinal < 0:
            raise ValueError("safety_margin_longitudinal must be >= 0")
        for name in ("direction_switch_penalty", "steer_weight", "steer_change_weight",
                     "clearance_weight", "safety_margin"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.max_gear_switches is not None and self.max_gear_switches < 0:
            raise ValueError("max_gear_switches must be >= 0 or None")
        if not (0 <= self.min_slot_maneuver_length <= self.min_maneuver_length <= self.preferred_maneuver_length):
            raise ValueError("need 0 <= min_slot_maneuver_length <= min_maneuver_length <= preferred_maneuver_length")
        if self.max_expanded_nodes <= 0 or self.max_planning_time <= 0:
            raise ValueError("search limits must be positive")
        if self.sample_ds >= self.obstacle_heuristic_resolution:
            raise ValueError("sample_ds must be smaller than obstacle_heuristic_resolution "
                             "(required by the unreachability proof)")
        self.primitive_config().validate()

    def primitive_config(self) -> PrimitiveConfig:
        return PrimitiveConfig(n_steer=self.n_steer, length_classes=tuple(self.length_classes),
                               yaw_resolution=self.yaw_resolution, xy_resolution=self.xy_resolution,
                               sample_ds=self.sample_ds, allow_reverse=self.allow_reverse,
                               generator=self.primitive_generator)

    def with_changes(self, **kwargs) -> "PlannerConfig":
        return replace(self, **kwargs)


@dataclass
class PathSegment:
    """One primitive or analytic piece of the final path (for analysis)."""
    source: str             # "lattice" | "analytic"
    direction: int
    steering: float
    length: float


@dataclass
class PlanResult:
    status: PlanStatus
    message: str = ""
    trajectory: Optional[Trajectory] = None
    planning_time: float = 0.0
    expanded_nodes: int = 0
    generated_nodes: int = 0
    collision_checks: int = 0
    exact_collision_checks: int = 0
    analytic_attempts: int = 0
    analytic_success: bool = False
    cost: float = math.inf
    explored: Optional[np.ndarray] = None
    segments: List[PathSegment] = field(default_factory=list)
    n_primitives: int = 0
    search: str = "forward"  # which search produced the result
    raw_trajectory: Optional[Trajectory] = None  # search output before post-processing

    @property
    def success(self) -> bool:
        return self.status == PlanStatus.SUCCESS


class StateDiscretizer:
    """Maps continuous (x, y, yaw, direction) to a lattice cell key.

    x/y are floored to ``xy_resolution`` cells relative to the map origin; yaw
    is rounded to the nearest multiple of ``yaw_resolution`` and wrapped, so
    that -pi and +pi map to the same index.
    """

    def __init__(self, xy_resolution: float, yaw_resolution: float, origin=(0.0, 0.0),
                 include_direction: bool = True):
        if xy_resolution <= 0 or yaw_resolution <= 0:
            raise ValueError("resolutions must be positive")
        self.xy_resolution = xy_resolution
        self.yaw_resolution = yaw_resolution
        self.n_yaw = int(round(2 * math.pi / yaw_resolution))
        self.origin = (float(origin[0]), float(origin[1]))
        self.include_direction = include_direction

    def xy_index(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor((x - self.origin[0]) / self.xy_resolution)),
                int(math.floor((y - self.origin[1]) / self.xy_resolution)))

    def xy_center(self, ix: int, iy: int) -> Tuple[float, float]:
        return (self.origin[0] + (ix + 0.5) * self.xy_resolution,
                self.origin[1] + (iy + 0.5) * self.xy_resolution)

    def yaw_index(self, yaw: float) -> int:
        return int(round(yaw / self.yaw_resolution)) % self.n_yaw

    def yaw_value(self, iyaw: int) -> float:
        return wrap_angle(iyaw * self.yaw_resolution)

    def key(self, x: float, y: float, yaw: float, direction: int = 0) -> tuple:
        ix, iy = self.xy_index(x, y)
        return (ix, iy, self.yaw_index(yaw), direction if self.include_direction else 0)

    def keys(self, poses: np.ndarray, directions: np.ndarray) -> List[tuple]:
        ix = np.floor((poses[:, 0] - self.origin[0]) / self.xy_resolution).astype(np.int64)
        iy = np.floor((poses[:, 1] - self.origin[1]) / self.xy_resolution).astype(np.int64)
        iyaw = np.round(poses[:, 2] / self.yaw_resolution).astype(np.int64) % self.n_yaw
        if self.include_direction:
            return list(zip(ix.tolist(), iy.tolist(), iyaw.tolist(), np.asarray(directions).tolist()))
        return list(zip(ix.tolist(), iy.tolist(), iyaw.tolist(), [0] * len(ix)))


def inter_sample_bound(vehicle: VehicleInfo, margin: float, sample_ds: float) -> float:
    """Upper bound of how far any point of the (inflated) footprint can be from
    the footprint at the nearest sample while moving between two samples.

    A body point at (x, y) in the rear-axle frame travels at most
    ``ds * sqrt((1 - k*y)^2 + (k*x)^2)`` for a rear-axle step ``ds`` with
    curvature |k| <= k_max; the maximum over a convex footprint is attained at a
    corner. Any point touched between two samples is within half of that
    distance of one of the two sampled footprints.
    """
    k = vehicle.max_curvature
    corners = vehicle.footprint_local(margin)
    travel = max(math.hypot(1.0 + k * abs(y), k * abs(x)) for x, y in corners)
    return 0.5 * sample_ds * travel


class _Nodes:
    """Structure-of-arrays node storage (fast, deterministic)."""

    __slots__ = ("x", "y", "yaw", "d", "st", "g", "parent", "prim", "key", "goal", "rs", "sw", "run", "cz", "sg")

    def __init__(self):
        self.x: List[float] = []
        self.y: List[float] = []
        self.yaw: List[float] = []
        self.d: List[int] = []
        self.st: List[float] = []
        self.g: List[float] = []
        self.parent: List[int] = []
        self.prim: List[int] = []
        self.key: List[Optional[tuple]] = []
        self.goal: List[bool] = []
        self.rs: Dict[int, tuple] = {}
        self.sw: List[int] = []      # gear changes so far
        self.run: List[float] = []   # distance driven in the current gear
        self.cz: List[bool] = []     # the last gear change happened near the slot
        self.sg: List[int] = []      # last non-zero steering sign in the current manoeuvre

    def add(self, x, y, yaw, d, st, g, parent, prim, key, goal=False, sw=0, run=math.inf, cz=True, sg=0) -> int:
        self.x.append(x); self.y.append(y); self.yaw.append(yaw); self.d.append(d)
        self.st.append(st); self.g.append(g); self.parent.append(parent); self.prim.append(prim)
        self.key.append(key); self.goal.append(goal); self.sw.append(sw); self.run.append(run)
        self.cz.append(cz); self.sg.append(sg)
        return len(self.x) - 1

    def __len__(self) -> int:
        return len(self.x)


class StateLatticePlanner:
    """Vehicle/configuration dependent part of the planner (built once)."""

    def __init__(self, vehicle: VehicleInfo, config: PlannerConfig = PlannerConfig()):
        vehicle.validate()
        config.validate()
        self.vehicle = vehicle
        self.config = config
        # regenerated automatically for any vehicle / lattice parameter change (cached)
        self.primitives: PrimitiveSet = generate_primitives(vehicle, config.primitive_config())
        self.max_steer = vehicle.max_steer_angle
        #: extra check margin so that the safety margins hold between samples too
        m = self.margins
        self.sweep_margin = inter_sample_bound(vehicle, m, config.sample_ds)
        #: (longitudinal, lateral) margins used by the collision checker
        self.check_margins = (m[0] + self.sweep_margin, m[1] + self.sweep_margin)
        self.check_margin = min(self.check_margins)
        # vehicle/primitive/cost dependent heuristic tables are built (or loaded from
        # the disk cache) here, i.e. offline w.r.t. the planning call
        self._hlut = self._build_lattice_table()

    # --------------------------------------------------------------- public
    def plan(self, grid: OccupancyGrid, start, goal) -> PlanResult:
        """Plan with the configured limits. Never raises."""
        t0 = time.perf_counter()
        search = self.start_search(grid, start, goal)
        res = search.result
        while res is None:
            res = search.step(256, deadline=t0 + self.config.max_planning_time)
        res.planning_time = time.perf_counter() - t0
        return res

    def start_search(self, grid: OccupancyGrid, start, goal, name: str = "forward",
                     extra_starts=(), max_gear_switches: Optional[int] = None,
                     max_expansions: Optional[int] = None, slot_pose=None) -> "LatticeSearch":
        """``extra_starts``: additional root poses with an initial cost
        [((x, y, yaw), g0), ...] (multi-source search, e.g. the whole goal
        tolerance region for the exit search). ``max_gear_switches``: tighter
        gear-change limit for this search (<= the configured one)."""
        return LatticeSearch(self, grid, start, goal, name, extra_starts, max_gear_switches, max_expansions,
                             slot_pose)

    def region_of_interest(self, grid: OccupancyGrid, start, goal) -> OccupancyGrid:
        m = self.config.roi_margin
        try:
            if m is None or not all(math.isfinite(float(v)) for v in tuple(start)[:2] + tuple(goal)[:2]):
                return grid
        except (TypeError, ValueError):
            return grid
        xs, ys = (start[0], goal[0]), (start[1], goal[1])
        return grid.crop(min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m)

    @property
    def margins(self) -> Tuple[float, float]:
        """Configured (longitudinal, lateral) safety margins."""
        cfg = self.config
        lon = cfg.safety_margin if cfg.safety_margin_longitudinal is None else cfg.safety_margin_longitudinal
        return (lon, cfg.safety_margin)

    def make_checker(self, grid: OccupancyGrid) -> CollisionChecker:
        cfg = self.config
        return CollisionChecker(grid, self.vehicle, cfg.collision_method, self.check_margins, cfg.n_circles)

    def make_zoned_checker(self, grid: OccupancyGrid, endpoints) -> Tuple[CollisionChecker, List[str]]:
        """Nominal checker, relaxed around endpoints that violate the nominal margin.

        Returns (checker, notes). Only the obstacle cells that are within the
        nominal check margin (+1 cell) of the endpoint footprint get the reduced
        margin, and only near that endpoint (see :class:`ZonedChecker`). An
        endpoint whose relaxed margin would fall below ``min_relaxed_margin`` is
        left unrelaxed, so it is still reported as in collision.
        """
        cfg = self.config
        nominal = self.make_checker(grid)
        if not cfg.endpoint_relaxation:
            return nominal, []
        radius = cfg.endpoint_relaxation_radius or self.vehicle.vehicle_length
        zones, notes = [], []
        for name, p in endpoints:
            if not nominal.check_pose(*p):
                continue
            clearance = float(nominal.min_clearance(np.array([p]), cap=self.check_margin + 1.0)[0])
            # check margin inside the zone: the endpoint's own clearance (5 mm slack);
            # the guaranteed continuous clearance there is m_check - sweep
            m_check = min(self.check_margin, clearance - 0.005)
            if m_check <= 0.0:
                continue
            guaranteed = m_check - inter_sample_bound(self.vehicle, m_check, cfg.sample_ds)
            if guaranteed < cfg.min_relaxed_margin:
                continue
            relaxed = CollisionChecker(grid, self.vehicle, cfg.collision_method, m_check, cfg.n_circles)
            if relaxed.check_pose(*p):
                continue
            others_grid = grid.copy()
            others_grid.data[self._constraining_cells(grid, p)] = 0
            others = CollisionChecker(others_grid, self.vehicle, cfg.collision_method, self.check_margins,
                                      cfg.n_circles)
            zones.append((p[0], p[1], radius, relaxed, others))
            notes.append(f"{name} margin relaxed to {guaranteed:.3f} m (only for the obstacles near it) "
                         f"within {radius:.1f} m")
        if not zones:
            return nominal, []
        return ZonedChecker(nominal, zones), notes

    def _constraining_cells(self, grid: OccupancyGrid, pose) -> Tuple[np.ndarray, np.ndarray]:
        """Occupied cells overlapping the footprint at ``pose`` inflated by the
        nominal check margin (+1 cell): the obstacles that make this endpoint tight."""
        from collision_checker import rect_square_distance
        v = self.vehicle
        reach = max(self.check_margins) + grid.resolution
        corners = v.footprint_world(*pose, reach + grid.resolution)
        ix0, iy0 = grid.world_to_index(corners[:, 0].min(), corners[:, 1].min())
        ix1, iy1 = grid.world_to_index(corners[:, 0].max(), corners[:, 1].max())
        ix0, iy0 = max(int(ix0), 0), max(int(iy0), 0)
        ix1, iy1 = min(int(ix1), grid.width - 1), min(int(iy1), grid.height - 1)
        sub = grid.data[iy0:iy1 + 1, ix0:ix1 + 1]
        jy, jx = np.nonzero(sub)
        if len(jx) == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        qx, qy = grid.index_to_world(jx + ix0, jy + iy0)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        fcx, fcy = v.footprint_center_offset
        pcx = pose[0] + c * fcx - s * fcy
        pcy = pose[1] + s * fcx + c * fcy
        # the check margin inflates the footprint as a RECTANGLE (corners reach
        # margin*sqrt(2)), so select the cells overlapping the inflated rectangle
        d = rect_square_distance(pcx, pcy, c, s, v.half_length + reach, v.half_width + reach,
                                 np.column_stack([qx, qy]), grid.resolution / 2)
        near = d <= 0.0
        return jy[near] + iy0, jx[near] + ix0

    def discretizer(self, grid: OccupancyGrid) -> StateDiscretizer:
        cfg = self.config
        return StateDiscretizer(cfg.xy_resolution, cfg.yaw_resolution, grid.origin,
                                cfg.include_direction_in_state)

    def state_key(self, x: float, y: float, yaw: float, d: int, grid: OccupancyGrid) -> tuple:
        return self.discretizer(grid).key(x, y, yaw, d)

    def lattice_table(self):
        return self._hlut

    def goal_reached(self, x, y, yaw, goal) -> bool:
        return (math.hypot(x - goal[0], y - goal[1]) <= self.config.position_tolerance
                and abs(wrap_angle(yaw - goal[2])) <= self.config.yaw_tolerance)

    def segment_cost(self, length, direction, steer, prev_dir, prev_steer, min_clearance) -> float:
        cfg = self.config
        c = length * (cfg.forward_weight if direction > 0 else cfg.reverse_weight)
        c += cfg.steer_weight * length * abs(steer) / self.max_steer
        if prev_dir != 0:
            c += cfg.steer_change_weight * abs(steer - prev_steer) / self.max_steer
            if direction != prev_dir:
                c += cfg.direction_switch_penalty
        if cfg.clearance_weight > 0 and cfg.clearance_ref > 0:
            c += cfg.clearance_weight * length * max(0.0, 1.0 - min_clearance / cfg.clearance_ref)
        return c

    def _build_lattice_table(self):
        """Free-space lattice cost-to-go table (None if unavailable / not requested)."""
        cfg = self.config
        if cfg.heuristic != "lattice_obstacle":
            return None
        from lattice_heuristic import LatticeHLUT, MultiResolutionHLUT
        if not LatticeHLUT.available():
            return None
        common = (self.primitives, cfg.reverse_weight, cfg.steer_weight, cfg.direction_switch_penalty)
        # one table level per number of remaining gear changes (0..max) if limited
        levels = cfg.max_gear_switches + 1 if cfg.max_gear_switches is not None else 0
        fine = LatticeHLUT.get(*common, cfg.hlut_extent, cfg.hlut_resolution, cfg.position_tolerance,
                               cfg.forward_weight, levels)
        if cfg.hlut_far_extent <= cfg.hlut_extent:
            return fine
        far = LatticeHLUT.get(*common, cfg.hlut_far_extent, cfg.hlut_far_resolution, cfg.position_tolerance,
                              cfg.forward_weight, levels)
        return MultiResolutionHLUT([fine, far])


class LatticeSearch:
    """One A* search. ``result`` is set as soon as the search terminates."""

    def __init__(self, planner: StateLatticePlanner, grid: OccupancyGrid, start, goal, name: str = "forward",
                 extra_starts=(), max_gear_switches: Optional[int] = None,
                 max_expansions: Optional[int] = None, slot_pose=None):
        self.planner = planner
        self._slot_pose = slot_pose
        self.on_expand = None  # optional callback(node_id) after each expansion (meet-in-the-middle)
        self.max_expansions = max_expansions or planner.config.max_expanded_nodes
        self.extra_starts = list(extra_starts)
        cap = planner.config.max_gear_switches
        if max_gear_switches is not None:
            cap = max_gear_switches if cap is None else min(cap, max_gear_switches)
        self.cap: Optional[int] = cap
        self.name = name
        self.result: Optional[PlanResult] = None
        self.t_start = time.perf_counter()
        self.expanded = 0
        self.analytic_attempts = 0
        self._analytic_fail_streak = 0
        self.explored: List[Tuple[float, float]] = []
        try:
            self._setup(grid, start, goal)
        except Exception as exc:  # never crash the caller
            self._error(exc)

    # ------------------------------------------------------------ set-up
    def _setup(self, grid, start, goal) -> None:
        pl, cfg = self.planner, self.planner.config
        try:
            start = tuple(float(v) for v in start)
            goal = tuple(float(v) for v in goal)
            if len(start) != 3 or len(goal) != 3 or not all(map(math.isfinite, start + goal)):
                raise ValueError
        except (TypeError, ValueError):
            self._terminate(PlanStatus.INVALID_INPUT, "start/goal must be finite (x, y, yaw)")
            return
        self.start = (start[0], start[1], wrap_angle(start[2]))
        self.goal = (goal[0], goal[1], wrap_angle(goal[2]))
        grid = pl.region_of_interest(grid, self.start, self.goal)
        self.grid = grid
        self.checker, self.relaxation_notes = pl.make_zoned_checker(
            grid, (("start", self.start), ("goal", self.goal)))
        if self.checker.check_pose(*self.start):
            self._terminate(PlanStatus.START_IN_COLLISION, "start pose collides (or is outside the map)")
            return
        if self.checker.check_pose(*self.goal):
            self._terminate(PlanStatus.GOAL_IN_COLLISION, "goal pose collides (or is outside the map)")
            return
        # the unreachability proof must use the smallest margin any pose is checked with
        self.heur, obs_h = build_heuristic(cfg.heuristic, grid, self.goal, pl.vehicle, self.checker.margin,
                                           self.checker.edt, cfg.obstacle_heuristic_resolution,
                                           pl.lattice_table(),
                                           point_robot=cfg.collision_method == "point")
        if obs_h is not None and not obs_h.reachable(self.start[0], self.start[1]):
            self._terminate(PlanStatus.UNREACHABLE, "goal not reachable even for the inflated 2D point model")
            return
        self.rules = ManeuverRules(cfg, pl.vehicle, self._slot_pose or self.goal, self.cap)
        self.rs_h = (ReedsSheppHeuristic(self.goal, pl.vehicle.min_turning_radius)
                     if cfg.use_analytic_expansion else None)
        self.disc = pl.discretizer(grid)
        self.nodes = _Nodes()
        self.open: List[tuple] = []
        self.best_g: Dict[tuple, float] = {}
        self.closed = set()
        self.counter = 0
        self.best_goal_cost = math.inf
        pset = pl.primitives
        self.base_cost = (pset.lengths * np.where(pset.directions > 0, cfg.forward_weight, cfg.reverse_weight)
                          + cfg.steer_weight * pset.lengths * np.abs(pset.steerings) / pl.max_steer)
        classes = np.array([p.length_class for p in pset.primitives])
        self._open_mask = np.isin(classes, ("medium", "long"))    # open space: coarse moves
        self._tight_mask = ~np.isin(classes, ("long",))            # near obstacles: fine moves
        if not self._open_mask.any():
            self._open_mask = np.ones(len(pset), dtype=bool)
        h0 = float(self.heur(np.array([self.start]), np.array([0]), self._remaining(0))[0])
        k0 = self._key(self.disc.key(*self.start, 0), 0, math.inf)
        self.root = self.nodes.add(*self.start, 0, 0.0, 0.0, -1, -1, k0)
        self.best_g[k0] = 0.0
        heapq.heappush(self.open, (cfg.heuristic_weight * h0, h0, 0, self.root))
        for pose, g0 in self.extra_starts:
            pose = (float(pose[0]), float(pose[1]), wrap_angle(float(pose[2])))
            if self.checker.check_pose(*pose):
                continue
            k = self._key(self.disc.key(*pose, 0), 0, math.inf)
            if g0 >= self.best_g.get(k, math.inf):
                continue
            h = float(self.heur(np.array([pose]), np.array([0]), self._remaining(0))[0])
            if not math.isfinite(h):
                continue
            self.best_g[k] = g0
            nid = self.nodes.add(*pose, 0, 0.0, float(g0), -1, -1, k)
            self.counter += 1
            heapq.heappush(self.open, (g0 + cfg.heuristic_weight * h, h, self.counter, nid))
        if pl.goal_reached(*self.start, self.goal):
            gid = self.nodes.add(*self.start, 0, 0.0, 0.0, self.root, -1, None, goal=True)
            self.result = self._reconstruct(gid)

    # ------------------------------------------------------ manoeuvre rules
    def _key(self, base: tuple, sw: int, run: float) -> tuple:
        """Lattice cell + gear (base) + what the hard manoeuvre constraints depend
        on: number of gear changes (if limited) and the current run length class."""
        return base + (sw if self.cap is not None else 0, self.rules.run_class(run))

    def _dominated(self, k: tuple, g: float) -> bool:
        """Dominated by a state in the same cell/gear with <= gear changes, a run
        class >= this one and <= cost (everything reachable from this state is
        reachable from that one)."""
        base, sw, rc = k[:4], k[4], k[5]
        best_g = self.best_g
        for s2 in range(sw + 1):
            for rc2 in range(rc, 3):
                if (s2, rc2) != (sw, rc) and best_g.get(base + (s2, rc2), math.inf) <= g:
                    return True
        return False

    def _remaining(self, sw):
        return None if self.cap is None else np.maximum(self.cap - np.asarray(sw), 0)

    # ------------------------------------------------------------ stepping
    def step(self, n: int = 256, deadline: Optional[float] = None) -> Optional[PlanResult]:
        """Run up to ``n`` expansions. Returns the result once terminated."""
        if self.result is not None:
            return self.result
        try:
            self._run(n, deadline)
        except Exception as exc:
            self._error(exc)
        return self.result

    def bound_cost(self, cost: float) -> None:
        """Only look for solutions cheaper than ``cost`` from now on (branch & bound
        with a solution found elsewhere, e.g. by the other search direction)."""
        if cost < self.best_goal_cost:
            self.best_goal_cost = cost

    def stop(self, status: PlanStatus, message: str) -> PlanResult:
        """Terminate from outside (e.g. shared time budget exhausted)."""
        if self.result is None:
            self._terminate(status, message)
        return self.result

    def _run(self, n: int, deadline: Optional[float]) -> None:
        cfg = self.planner.config
        nodes = self.nodes
        for _ in range(n):
            if not self.open:
                self._terminate(PlanStatus.NO_PATH, "open list exhausted")
                return
            if self.expanded >= self.max_expansions:
                self._terminate(PlanStatus.MAX_EXPANSIONS, f"expanded {self.expanded} nodes")
                return
            if deadline is not None and (self.expanded & 31) == 0 and time.perf_counter() > deadline:
                self._terminate(PlanStatus.TIMEOUT, f"exceeded {cfg.max_planning_time}s")
                return
            _, _, _, nid = heapq.heappop(self.open)
            if nodes.goal[nid]:
                self.result = self._reconstruct(nid)
                return
            key = nodes.key[nid]
            if key in self.closed or nodes.g[nid] > self.best_g.get(key, math.inf) + 1e-9:
                continue
            self.closed.add(key)
            self.expanded += 1
            if cfg.record_explored:
                self.explored.append((nodes.x[nid], nodes.y[nid]))
            self._goal_checks(nid)
            self._expand(nid)
            if self.on_expand is not None:
                self.on_expand(nid)

    def _push_goal(self, parent: int, pose, d: int, st: float, cost: float, rs=None) -> None:
        gid = self.nodes.add(pose[0], pose[1], pose[2], d, st, cost, parent, -3 if rs else -2, None, goal=True)
        if rs is not None:
            self.nodes.rs[gid] = rs
        self.best_goal_cost = cost
        self.counter += 1
        heapq.heappush(self.open, (cost, 0.0, self.counter, gid))

    def _goal_checks(self, nid: int) -> None:
        pl, cfg, nodes = self.planner, self.planner.config, self.nodes
        x, y, yaw, d, st, g = nodes.x[nid], nodes.y[nid], nodes.yaw[nid], nodes.d[nid], nodes.st[nid], nodes.g[nid]
        sw, run, cz, sg = nodes.sw[nid], nodes.run[nid], nodes.cz[nid], nodes.sg[nid]
        # lattice goal (within tolerance); the final run must be a real manoeuvre
        if nodes.parent[nid] != -1 and pl.goal_reached(x, y, yaw, self.goal):
            ok, soft = self.rules.final_ok(d, run, sw, cz)
            if ok and g + soft < self.best_goal_cost:
                self._push_goal(nid, (x, y, yaw), d, st, g + soft)
        # analytic expansion (exact goal)
        if self.rs_h is None:
            return
        h_rs = float(self.rs_h(np.array([[x, y, yaw]]))[0])
        # deterministic back-off: after many consecutive failed shots, try only
        # every k-th expansion (k grows up to analytic_max_backoff)
        backoff = min(1 + self._analytic_fail_streak // 16, cfg.analytic_max_backoff)
        near = h_rs <= cfg.analytic_near_distance and self.expanded % backoff == 0
        if not ((near or self.expanded % cfg.analytic_interval == 1) and g + h_rs < self.best_goal_cost):
            return
        self.analytic_attempts += 1
        shot = self._analytic(x, y, yaw, d, st, self.best_goal_cost - g, sw, run, cz, sg)
        self._analytic_fail_streak = 0 if shot is not None else self._analytic_fail_streak + 1
        if shot is not None:
            cost, path, samples = shot
            self._push_goal(nid, self.goal, 0, 0.0, g + cost, (path, samples))

    def _expand(self, nid: int) -> None:
        pl, cfg, nodes = self.planner, self.planner.config, self.nodes
        pset = pl.primitives
        P = len(pset)
        dirs, steers, lengths = pset.directions, pset.steerings, pset.lengths
        x, y, yaw, d, st, g = nodes.x[nid], nodes.y[nid], nodes.yaw[nid], nodes.d[nid], nodes.st[nid], nodes.g[nid]
        sw, run, cz, sg = nodes.sw[nid], nodes.run[nid], nodes.cz[nid], nodes.sg[nid]
        key = nodes.key[nid]
        rules = self.rules
        # gear-change rules (hard): manoeuvre long enough (depends on the place), <= cap changes
        switching = (dirs != d) if d != 0 else np.zeros(P, dtype=bool)
        near = rules.near_slot(x, y)
        allowed = ~switching
        if d != 0 and rules.can_switch(run, sw, near):
            allowed = np.ones(P, dtype=bool)
        new_sw = sw + switching.astype(np.int64)
        new_run = np.where(switching | (d == 0), lengths, run + lengths)
        signs = np.sign(steers).astype(np.int64)
        new_cz = np.where(switching, near, cz)
        new_sg = np.where(switching | (signs != 0), signs, sg)
        reversal = ~switching & (sg != 0) & (signs != 0) & (signs != sg)
        ends = pset.transform_ends(x, y, yaw)
        base_keys = self.disc.keys(ends, dirs)
        keys = [self._key(bk, int(a), float(r)) for bk, a, r in zip(base_keys, new_sw, new_run)]
        closed = self.closed
        mask = allowed & np.fromiter((k not in closed and k != key for k in keys), dtype=bool, count=P)
        if cfg.open_space_clearance is not None:
            clr0 = float(self.checker.clearance_estimate(np.array([[x, y, yaw]]))[0])
            mask &= self._open_mask if clr0 >= cfg.open_space_clearance else self._tight_mask
        if cfg.max_steer_change is not None and d != 0:
            mask &= (dirs != d) | (np.abs(steers - st) <= cfg.max_steer_change + 1e-9)
        if not mask.any():
            return
        poses, owner = pset.transform_samples(x, y, yaw, mask)
        coll = self.checker.check_poses(poses)
        valid = mask & ~(np.bincount(owner, weights=coll, minlength=P) > 0)
        if not valid.any():
            return
        cost = self.base_cost.copy()
        if d != 0:
            cost += cfg.steer_change_weight * np.abs(steers - st) / pl.max_steer
            cost += cfg.direction_switch_penalty * switching
            cost += switching * rules.switch_cost(run)                  # soft: short manoeuvre
            cost += reversal * cfg.steer_reversal_penalty               # soft: steering wobble
        if cfg.clearance_weight > 0 and cfg.clearance_ref > 0:
            clr = self.checker.clearance_estimate(poses)
            prim_clr = np.full(P, np.inf)
            np.minimum.at(prim_clr, owner, clr)
            cost += cfg.clearance_weight * lengths * np.maximum(0.0, 1.0 - prim_clr / cfg.clearance_ref)
        vidx = np.nonzero(valid)[0]
        hs = self.heur(ends[vidx], dirs[vidx], self._remaining(new_sw[vidx]))
        w = cfg.heuristic_weight
        for j, h in zip(vidx, hs):
            if not math.isfinite(h):
                continue  # provably cannot reach the goal from here
            k = keys[j]
            g_new = g + float(cost[j])
            if g_new >= self.best_g.get(k, math.inf) or g_new >= self.best_goal_cost:
                continue
            if self._dominated(k, g_new):
                continue
            self.best_g[k] = g_new
            e = ends[j]
            cid = nodes.add(float(e[0]), float(e[1]), float(e[2]), int(dirs[j]), float(steers[j]),
                            g_new, nid, int(j), k, sw=int(new_sw[j]), run=float(new_run[j]),
                            cz=bool(new_cz[j]), sg=int(new_sg[j]))
            self.counter += 1
            heapq.heappush(self.open, (g_new + w * float(h), float(h), self.counter, cid))

    # --------------------------------------------------------- analytic shot
    def _analytic(self, x, y, yaw, d, st, budget: float, sw: int = 0, run: float = math.inf,
                  cz: bool = True, sg: int = 0):
        pl, cfg = self.planner, self.planner.config
        paths = rs_paths((x, y, yaw), self.goal, pl.vehicle.min_turning_radius)
        scored = []
        for p in paths:
            if p.shortest_segment() < cfg.min_rs_segment_length:
                continue  # a few-cm arc between two others is a steering spike
            ok, soft = self.rules.evaluate(p.pieces((x, y, yaw)), x, y, d, run, sw, sg, cz, final=True)
            if not ok:
                continue  # too many gear changes or a too-short manoeuvre
            c, pd, pst = soft, d, st
            for t, l in p.segments():
                sd = 1 if l > 0 else -1
                sst = pl.max_steer if t == "L" else (-pl.max_steer if t == "R" else 0.0)
                c += pl.segment_cost(abs(l), sd, sst, pd, pst, math.inf)
                pd, pst = sd, sst
            if c < budget:
                scored.append((c, p))
        scored.sort(key=lambda t: t[0])
        for c, p in scored[:cfg.analytic_max_candidates]:
            # cheap coarse rejection first (most candidates collide), then full resolution
            coarse, _, _, _ = sample_rs_path(p, (x, y, yaw), max(0.3, cfg.sample_ds))
            if self.checker.any_collision(coarse):
                continue
            poses, sg, sd, seg = sample_rs_path(p, (x, y, yaw), cfg.sample_ds)
            if len(poses) == 0 or self.checker.any_collision(poses):
                continue
            if cfg.clearance_weight > 0 and cfg.clearance_ref > 0:
                clr = self.checker.clearance_estimate(poses)
                seg_len = np.array([abs(l) for _, l in p.segments()])
                seg_min = np.full(len(seg_len), np.inf)
                np.minimum.at(seg_min, seg, clr)
                c += float(np.sum(cfg.clearance_weight * seg_len *
                                  np.maximum(0.0, 1.0 - seg_min / cfg.clearance_ref)))
                if c >= budget:
                    continue
            return c, p, (poses, sg * pl.max_steer, sd)
        return None

    # ------------------------------------------------------ termination
    def _stats(self) -> dict:
        checker = getattr(self, "checker", None)
        nodes = getattr(self, "nodes", None)
        return dict(expanded_nodes=self.expanded, generated_nodes=len(nodes) if nodes is not None else 0,
                    collision_checks=checker.check_count if checker else 0,
                    exact_collision_checks=checker.exact_count if checker else 0,
                    analytic_attempts=self.analytic_attempts,
                    explored=np.array(self.explored) if self.explored else np.zeros((0, 2)),
                    n_primitives=len(self.planner.primitives), search=self.name,
                    planning_time=time.perf_counter() - self.t_start)

    def _terminate(self, status: PlanStatus, message: str) -> None:
        self.result = PlanResult(status, message, **self._stats())

    def _error(self, exc: Exception) -> None:
        self.result = PlanResult(PlanStatus.ERROR, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                                 **self._stats())

    def _final_collision(self, poses: np.ndarray) -> bool:
        """Re-check the returned path with the *other* implementation of the
        cascade (numpy if the search used numba) - diverse redundancy rather
        than re-running the identical code."""
        chk = self.checker
        if isinstance(chk, ZonedChecker):
            return bool(chk.check_poses(poses).any())
        alt = CollisionChecker(self.grid, chk.vehicle, chk.method, chk.margin, chk.n_circles,
                               use_numba=not chk.use_numba)
        return bool(alt.check_poses(poses).any() or chk.check_poses(poses).any())

    def inject_goal(self, parent: int, final_pose, cost: float, tail) -> None:
        """Add a complete solution found outside this search: the path of
        ``parent`` followed by ``tail`` = (poses, steering, directions) arrays."""
        if cost < self.best_goal_cost:
            self._push_goal(parent, final_pose, 0, 0.0, cost, (None, tail))

    def branch(self, nid: int):
        """(poses, directions, steering) of the path from the root to node ``nid``."""
        chain = []
        n = nid
        while n != -1:
            chain.append(n)
            n = self.nodes.parent[n]
        chain.reverse()
        return self._chain_arrays(chain)[:3]

    def _chain_arrays(self, chain):
        pl, nodes = self.planner, self.nodes
        pset = pl.primitives
        pose_parts, dir_parts, steer_parts = [], [], []
        segments: List[PathSegment] = []
        analytic = False
        for n in chain[1:]:
            if n in nodes.rs:
                path, (poses, stv, dv) = nodes.rs[n]
                pose_parts.append(poses)
                dir_parts.append(dv)
                steer_parts.append(stv)
                if path is not None:
                    for t, l in path.segments():
                        segments.append(PathSegment("analytic", 1 if l > 0 else -1,
                                                    pl.max_steer if t == "L" else (-pl.max_steer if t == "R" else 0.0),
                                                    abs(l)))
                analytic = True
            elif nodes.prim[n] >= 0:
                p = nodes.prim[n]
                par = nodes.parent[n]
                prim = pset.primitives[p]
                poses, _ = pset.transform_samples(nodes.x[par], nodes.y[par], nodes.yaw[par],
                                                  np.arange(len(pset)) == p)
                pose_parts.append(poses)
                dir_parts.append(np.full(len(poses), prim.direction))
                steer_parts.append(np.full(len(poses), prim.steering))
                segments.append(PathSegment("lattice", prim.direction, prim.steering, prim.length))
        root = chain[0]  # the start, or one of the extra roots
        root_pose = np.array([[nodes.x[root], nodes.y[root], nodes.yaw[root]]])
        if pose_parts:
            poses = np.vstack([root_pose] + pose_parts)
            dirs = np.concatenate([[dir_parts[0][0]]] + dir_parts)
            steer = np.concatenate([[steer_parts[0][0]]] + steer_parts)
        else:
            poses, dirs, steer = root_pose, np.array([1]), np.array([0.0])
        return poses, dirs, steer, segments, analytic

    def _reconstruct(self, gid: int) -> PlanResult:
        pl, cfg, nodes = self.planner, self.planner.config, self.nodes
        chain = []
        n = gid
        while n != -1:
            chain.append(n)
            n = nodes.parent[n]
        chain.reverse()
        poses, dirs, steer, segments, analytic = self._chain_arrays(chain)
        traj = Trajectory.from_arrays(poses, dirs, steer, pl.vehicle.wheel_base)
        met = any(n in nodes.rs and nodes.rs[n][0] is None for n in chain)
        status, msg = PlanStatus.SUCCESS, ("meet-in-the-middle" if met else
                                           "analytic expansion" if analytic else "lattice goal")
        if self.relaxation_notes:
            msg += " (" + "; ".join(self.relaxation_notes) + ")"
        if cfg.final_validation and self._final_collision(poses):
            status, msg = PlanStatus.ERROR, "internal inconsistency: final trajectory collides"
        return PlanResult(status, msg, trajectory=traj, analytic_success=analytic, cost=nodes.g[gid],
                          segments=segments, **self._stats())
