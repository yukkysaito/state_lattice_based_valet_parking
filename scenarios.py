"""Evaluation scenarios for the state-lattice parking planner.

Every scenario is a fully rasterised :class:`OccupancyGrid` plus start / goal
poses (rear-axle center, Autoware ``base_link``), the vehicle parameters, the
planner-config overrides and an :class:`Expectation` that the test runner
checks. Scenarios are deterministic: :func:`build_scenario` rebuilds any case
from its id alone (random ``N-###`` cases from ``seed + index``), which is what
multiprocessing workers use.

Scene conventions
-----------------
* map resolution 0.1 m, origin (0, 0), maps kept compact (<= ~45 x 30 m);
* parked cars are 4.70 x 1.85 m rectangles;
* the default vehicle (Autoware default) is 4.69 x 1.90 m, WB 2.79 m,
  R_min = 4.55 m; the planner inflates the footprint by ``safety_margin``
  (0.1 m per side), i.e. the checked rectangle is 4.89 x 2.10 m;
* perpendicular lots: the slot row is at the bottom of the map (slots open
  towards +y), the aisle runs along x, the car starts in the aisle heading +x
  and has to reverse into the slot (goal yaw = +90 deg). The slot row is
  therefore on the *right* of the driving direction;
* parallel lots: curb at the bottom, the space is on the right of a car
  driving towards +x, goal yaw = 0;
* nothing outside the map is free (the planner treats it as collision), so a
  border wall is only added where it is physically meaningful.

Categories: see :data:`CATEGORIES`. Scene builders (:class:`Scene`,
:func:`perpendicular_lot`, :func:`parallel_lot`, :func:`open_area`) and
:func:`mirror_scenario` are reusable for new cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from occupancy_grid import OccupancyGrid, rectangle_polygon
from state_lattice import PlannerConfig
from vehicle import (VehicleInfo, arc_samples, default_vehicle, large_suv, long_wheelbase,
                     reduced_steer_sedan, small_car, standard_sedan, wrap_angle)

RESOLUTION = 0.1
DEFAULT_MARGIN = 0.1          # PlannerConfig().safety_margin
CAR_LENGTH = 4.70             # parked car footprint
CAR_WIDTH = 1.85
HALF_PI = math.pi / 2.0

CATEGORIES: Dict[str, Tuple[str, str]] = {
    "basic": ("A", "Basic"),
    "perpendicular": ("B", "Perpendicular"),
    "parallel": ("C", "Parallel"),
    "maneuver": ("D", "Multi-maneuver"),
    "obstacle": ("E", "Obstacle avoidance"),
    "collision": ("F", "Collision stress"),
    "unreachable": ("G", "Invalid input / unreachable"),
    "vehicle": ("H", "Vehicle variants"),
    "resolution": ("I", "Resolution sensitivity"),
    "primitive": ("J", "Primitive sensitivity"),
    "cost": ("K", "Cost tuning"),
    "tolerance": ("L", "Goal tolerance"),
    "symmetry": ("M", "Left/right symmetry"),
    "random": ("N", "Randomized"),
}
_LETTER_TO_CATEGORY = {v[0]: k for k, v in CATEGORIES.items()}


# =============================================================== data model
@dataclass
class Expectation:
    success: Optional[bool] = True        # None = don't care (parameter sweeps)
    expected_status: Optional[str] = None  # e.g. "start_in_collision", "goal_in_collision"
    expect_exception: bool = False        # building the vehicle must raise InvalidVehicleParameterError
    max_direction_changes: Optional[int] = None
    min_direction_changes: Optional[int] = None
    allowed_directions: Optional[str] = None  # "forward" | "reverse" | None
    max_time: Optional[float] = None      # loose planning-time bound [s] (must terminate)
    point_robot_should_succeed: bool = False  # a point-robot collision model would (wrongly) accept a path
    notes: str = ""


@dataclass
class Obstacle:
    polygon: np.ndarray   # (N, 2) world polygon, for visualization only
    kind: str             # "vehicle" | "wall" | "pillar" | "obstacle" | "curb"


@dataclass
class Scenario:
    id: str
    name: str
    category: str
    description: str
    grid: OccupancyGrid
    start: tuple
    goal: tuple
    vehicle_params: dict
    planner_overrides: dict = field(default_factory=dict)
    expectation: Expectation = field(default_factory=Expectation)
    obstacles: list = field(default_factory=list)
    group: str = ""
    tags: tuple = ()
    difficulty: str = ""      # "easy" | "normal" | "hard" (benchmark grouping); set by difficulty_of()
    # product specification (see spec_of): allowed gear changes, and whether a
    # fast rejection is an acceptable answer (slot too tight for the limits)
    spec_max_switches: Optional[int] = None
    spec_reject_ok: bool = False
    meta: dict = field(default_factory=dict)   # geometry of random cases (kind, ratio, pitch, aisle)

    @property
    def is_test(self) -> bool:
        """True if the scenario has a defined expected behaviour (pass/fail test).
        Scenarios without one (sweeps, random, open-ended hard cases) are only
        used for the performance benchmark."""
        e = self.expectation
        return e.success is not None or e.expected_status is not None or e.expect_exception

    def make_vehicle(self) -> VehicleInfo:
        """Build the vehicle (raises InvalidVehicleParameterError for G6)."""
        return VehicleInfo(**self.vehicle_params)

    def make_config(self, base: PlannerConfig = PlannerConfig()) -> PlannerConfig:
        return base.with_changes(**self.planner_overrides) if self.planner_overrides else base

    @property
    def letter(self) -> str:
        return CATEGORIES[self.category][0]

    def __repr__(self) -> str:  # compact (the grid is large)
        return (f"Scenario({self.id!r}, {self.name!r}, start={tuple(round(v, 3) for v in self.start)}, "
                f"goal={tuple(round(v, 3) for v in self.goal)}, map={self.grid.width}x{self.grid.height})")


def vehicle_params_of(v: VehicleInfo) -> dict:
    """Picklable ``VehicleInfo(**params)`` kwargs of a vehicle."""
    return {f.name: getattr(v, f.name) for f in fields(VehicleInfo)}


# ============================================================ scene builder
@dataclass
class _Shape:
    kind: str
    polygon: np.ndarray                              # drawing polygon (also raster polygon)
    circle: Optional[Tuple[float, float, float]] = None  # (cx, cy, r) -> rasterised as exact disc


class Scene:
    """Collects obstacle shapes in world coordinates and rasterises them.

    Keeping the geometry as shapes (instead of drawing directly into a grid)
    makes it cheap to transform a whole scene (mirror / rotate for random
    cases) before rasterisation.
    """

    def __init__(self, width: float, height: float, resolution: float = RESOLUTION):
        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)
        self.shapes: List[_Shape] = []

    # ------------------------------------------------------------ shapes
    def polygon(self, poly, kind: str = "obstacle") -> "Scene":
        self.shapes.append(_Shape(kind, np.asarray(poly, dtype=float)))
        return self

    def box(self, xmin: float, ymin: float, xmax: float, ymax: float, kind: str = "obstacle") -> "Scene":
        return self.polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], kind)

    def rect(self, cx: float, cy: float, length: float, width: float, yaw: float = 0.0,
             kind: str = "obstacle") -> "Scene":
        return self.polygon(rectangle_polygon(cx, cy, length, width, yaw), kind)

    def car(self, cx: float, cy: float, yaw: float = 0.0, length: float = CAR_LENGTH,
            width: float = CAR_WIDTH) -> "Scene":
        """Parked car centred at (cx, cy) with heading ``yaw``."""
        return self.rect(cx, cy, length, width, yaw, "vehicle")

    def circle(self, cx: float, cy: float, r: float, kind: str = "pillar") -> "Scene":
        t = np.linspace(0.0, 2 * math.pi, 33)[:-1]
        poly = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])
        self.shapes.append(_Shape(kind, poly, (float(cx), float(cy), float(r))))
        return self

    def enclosure(self, xmin: float, ymin: float, xmax: float, ymax: float, t: float = 0.3,
                  kind: str = "wall") -> "Scene":
        """Four walls of thickness ``t`` whose *inner* faces are the given box."""
        self.box(xmin - t, ymin - t, xmax + t, ymin, kind)
        self.box(xmin - t, ymax, xmax + t, ymax + t, kind)
        self.box(xmin - t, ymin, xmin, ymax, kind)
        self.box(xmax, ymin, xmax + t, ymax, kind)
        return self

    # ------------------------------------------------------- transforms
    def transformed(self, rot: np.ndarray, trans, width: float, height: float) -> "Scene":
        """New scene with every point p mapped to rot @ p + trans."""
        out = Scene(width, height, self.resolution)
        trans = np.asarray(trans, dtype=float)
        for s in self.shapes:
            poly = s.polygon @ rot.T + trans
            if np.linalg.det(rot) < 0:
                poly = poly[::-1].copy()
            circ = None
            if s.circle is not None:
                c = rot @ np.array(s.circle[:2]) + trans
                circ = (float(c[0]), float(c[1]), s.circle[2])
            out.shapes.append(_Shape(s.kind, poly, circ))
        return out

    # ----------------------------------------------------------- output
    def rasterize(self) -> OccupancyGrid:
        grid = OccupancyGrid.empty(self.width, self.height, self.resolution)
        for s in self.shapes:
            if s.circle is not None:
                grid.add_circle(*s.circle)
            else:
                grid.add_polygon(s.polygon)
        return grid

    def obstacles(self) -> List[Obstacle]:
        return [Obstacle(s.polygon.copy(), s.kind) for s in self.shapes]


def _transform_pose(pose, rot: np.ndarray, trans) -> tuple:
    p = rot @ np.array(pose[:2], dtype=float) + np.asarray(trans, dtype=float)
    h = rot @ np.array([math.cos(pose[2]), math.sin(pose[2])])
    return (float(p[0]), float(p[1]), float(math.atan2(h[1], h[0])))


def _make(sid: str, name: str, category: str, description: str, scene: Scene, start, goal,
          vehicle: Optional[VehicleInfo] = None, *, vehicle_params: Optional[dict] = None,
          overrides: Optional[dict] = None, expectation: Optional[Expectation] = None,
          group: str = "", tags: Sequence[str] = ()) -> Scenario:
    if vehicle_params is None:
        vehicle_params = vehicle_params_of(vehicle or default_vehicle())
    return Scenario(id=sid, name=name, category=category, description=description,
                    grid=scene.rasterize(), start=tuple(float(v) for v in start),
                    goal=tuple(float(v) for v in goal), vehicle_params=dict(vehicle_params),
                    planner_overrides=dict(overrides or {}),
                    expectation=expectation or Expectation(), obstacles=scene.obstacles(),
                    group=group, tags=tuple(tags))


# ============================================================ scene library
def open_area(width: float = 40.0, height: float = 20.0) -> Scene:
    """Empty scene (the map edge is the only boundary)."""
    return Scene(width, height)


@dataclass
class LotInfo:
    """Geometry summary returned by the lot builders (useful for new cases)."""
    goal_x: float
    slot_width: float
    curb_top: float
    aisle_y0: float
    aisle_y1: float
    slot_centers: List[float]
    occupied: List[float]


def perpendicular_lot(vehicle: Optional[VehicleInfo] = None, *, slot_width: float = 2.5,
                      aisle_width: float = 6.5, slot_depth: float = 5.0, neighbours="full",
                      opposite: Optional[str] = "cars", map_width: float = 36.0, goal_x: float = 18.0,
                      start_dx: float = -7.0, start_dy: float = 0.0, start_yaw: float = 0.0,
                      rear_gap: float = 0.3, dead_end_x: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None, jitter: float = 0.0):
    """Reverse perpendicular parking lot.

    Layout (y up): curb (y 0.2-0.4) | slot row, slots ``slot_width`` pitch and
    ``slot_depth`` deep | aisle ``aisle_width`` | opposite row of parked cars
    (``opposite="cars"``), a wall (``"wall"``) or nothing.

    ``neighbours``: "full" (every other slot of the row occupied), "none",
    an int n (n cars on each side) or a tuple (n_minus_x, n_plus_x).
    ``dead_end_x``: closes the aisle with a wall at that x (slots beyond it
    are dropped). ``jitter`` (with ``rng``): random offset amplitude of parked
    cars [m] (yaw jitter = 10 deg per metre of amplitude).

    Goal: rear axle ``rear_gap`` in front of the curb plus the rear overhang,
    yaw +90 deg (reversed in). Start: aisle centre (+``start_dy``), at
    ``goal_x + start_dx``, yaw ``start_yaw``.

    Returns (scene, start, goal, LotInfo).
    """
    v = vehicle or default_vehicle()
    curb_top = 0.4
    aisle_y0 = curb_top + slot_depth
    aisle_y1 = aisle_y0 + aisle_width
    if opposite == "cars":
        height = aisle_y1 + 0.3 + CAR_LENGTH + 0.3 + 0.2 + 0.3
    elif opposite == "wall":
        height = aisle_y1 + 0.3 + 0.2
    else:
        height = aisle_y1 + 0.3
    width = map_width if dead_end_x is None else dead_end_x + 0.3
    sc = Scene(width, height)
    xmax_slots = width if dead_end_x is None else dead_end_x
    sc.box(0.0, 0.2, width, curb_top, "curb")
    if dead_end_x is not None:
        sc.box(dead_end_x, 0.0, dead_end_x + 0.3, height, "wall")

    def jit():
        if rng is None or jitter <= 0:
            return 0.0, 0.0, 0.0
        return (float(rng.uniform(-jitter, jitter)), float(rng.uniform(-jitter, jitter)),
                float(rng.uniform(-1, 1)) * math.radians(10.0) * jitter)

    kmin = -int(math.floor((goal_x - slot_width / 2) / slot_width))
    kmax = int(math.floor((xmax_slots - goal_x - slot_width / 2) / slot_width))
    centers = [goal_x + k * slot_width for k in range(kmin, kmax + 1)]
    if neighbours == "full":
        nm, npl = 10 ** 6, 10 ** 6
    elif neighbours in ("none", None, 0):
        nm, npl = 0, 0
    elif isinstance(neighbours, int):
        nm, npl = neighbours, neighbours
    else:
        nm, npl = neighbours
    occupied = []
    cy = curb_top + rear_gap + CAR_LENGTH / 2
    for k in range(kmin, kmax + 1):
        if k == 0 or (k < 0 and -k > nm) or (k > 0 and k > npl):
            continue
        x = goal_x + k * slot_width
        if x - CAR_WIDTH / 2 < 0 or x + CAR_WIDTH / 2 > xmax_slots:
            continue
        dx, dy, dyaw = jit()
        sc.car(x + dx, cy + dy, HALF_PI + dyaw)
        occupied.append(x)
    if opposite == "cars":
        ocy = aisle_y1 + 0.3 + CAR_LENGTH / 2
        k = 0
        x = goal_x + slot_width / 2          # offset row (more realistic, not mirror-aligned)
        xs = []
        while x - slot_width * k >= CAR_WIDTH / 2:
            xs.append(x - slot_width * k)
            k += 1
        k = 1
        while x + slot_width * k <= xmax_slots - CAR_WIDTH / 2:
            xs.append(x + slot_width * k)
            k += 1
        for x in sorted(xs):
            dx, dy, dyaw = jit()
            sc.car(x + dx, ocy + dy, -HALF_PI + dyaw)
        top = aisle_y1 + 0.3 + CAR_LENGTH + 0.3
        sc.box(0.0, top, width, top + 0.2, "curb")
    elif opposite == "wall":
        sc.box(0.0, aisle_y1, width, aisle_y1 + 0.2, "wall")
    goal = (goal_x, curb_top + rear_gap + v.rear_overhang, HALF_PI)
    start = (goal_x + start_dx, 0.5 * (aisle_y0 + aisle_y1) + start_dy, start_yaw)
    info = LotInfo(goal_x, slot_width, curb_top, aisle_y0, aisle_y1, centers, occupied)
    return sc, start, goal, info


def parallel_lot(vehicle: Optional[VehicleInfo] = None, *, space_length: float = 7.0,
                 road_width: float = 6.0, map_width: float = 40.0, space_x: float = 16.0,
                 n_cars: Tuple[int, int] = (2, 2), car_gap: float = 1.2, curb_gap: float = 0.25,
                 start_dx: float = 1.0, start_gap: float = 1.0, start_yaw: float = 0.0,
                 rng: Optional[np.random.Generator] = None, jitter: float = 0.0):
    """Parallel parking on the right-hand curb of a road running along +x.

    Curb at y 0.2-0.4, parked cars 0.25 m from the curb, a space of
    ``space_length`` (bumper to bumper) starting at ``space_x``, ``n_cars``
    (behind, in front) cars with ``car_gap`` between them, and an opposite curb
    ``road_width`` beyond the outer side of the parked cars.

    Goal: vehicle centred in the space, ``curb_gap`` from the curb, yaw 0.
    Start: rear axle ``start_dx`` ahead of the space end (1.0 = rear bumper
    aligned with the front car's rear bumper for the default vehicle), body side
    ``start_gap`` from the parked cars, yaw ``start_yaw``.

    Returns (scene, start, goal, info dict).
    """
    v = vehicle or default_vehicle()
    curb_top = 0.4
    lane_edge = curb_top + curb_gap + CAR_WIDTH
    opp = lane_edge + road_width
    height = opp + 0.2 + 0.3
    sc = Scene(map_width, height)
    sc.box(0.0, 0.2, map_width, curb_top, "curb")
    sc.box(0.0, opp, map_width, opp + 0.2, "curb")
    cy = curb_top + curb_gap + CAR_WIDTH / 2

    def jit():
        if rng is None or jitter <= 0:
            return 0.0, 0.0, 0.0
        return (float(rng.uniform(-jitter, jitter)), float(rng.uniform(-jitter, jitter) * 0.5),
                float(rng.uniform(-1, 1)) * math.radians(6.0) * jitter)

    pitch = CAR_LENGTH + car_gap
    for k in range(n_cars[0]):
        x = space_x - CAR_LENGTH / 2 - k * pitch
        dx, dy, dyaw = jit() if k else (0.0, *jit()[1:])   # the space itself is not shrunk by jitter
        sc.car(x + dx, cy + dy, dyaw)
    for k in range(n_cars[1]):
        x = space_x + space_length + CAR_LENGTH / 2 + k * pitch
        dx, dy, dyaw = jit() if k else (0.0, *jit()[1:])
        sc.car(x + dx, cy + dy, dyaw)
    goal = (space_x + space_length / 2 - (v.vehicle_length / 2 - v.rear_overhang),
            curb_top + curb_gap + v.half_width, 0.0)
    start = (space_x + space_length + start_dx, lane_edge + start_gap + v.half_width, start_yaw)
    info = dict(space_x=space_x, space_length=space_length, lane_edge=lane_edge, road_y1=opp)
    return sc, start, goal, info


# ================================================================ mirroring
def mirror_scenario(sc: Scenario, axis: str = "horizontal", new_id: Optional[str] = None,
                    name: Optional[str] = None, group: Optional[str] = None) -> Scenario:
    """Exact left/right mirror of a scenario.

    ``axis="horizontal"``: mirror about the map's horizontal centre line
    (y -> H - y, yaw -> -yaw); a car driving +x sees left and right swapped.
    ``axis="vertical"``: mirror about the vertical centre line
    (x -> W - x, yaw -> pi - yaw).
    The grid is flipped cell-exactly, so the mirrored problem is exactly the
    mirror image (the vehicle must be left/right symmetric, which all presets
    are).
    """
    g = sc.grid
    xmin, xmax, ymin, ymax = g.extent
    if axis == "horizontal":
        c = 0.5 * (ymin + ymax)
        data = g.data[::-1, :].copy()

        def mp(p):
            return (p[0], 2 * c - p[1], wrap_angle(-p[2]))

        def mpoly(poly):
            out = poly.copy()
            out[:, 1] = 2 * c - out[:, 1]
            return out[::-1].copy()
    elif axis == "vertical":
        c = 0.5 * (xmin + xmax)
        data = g.data[:, ::-1].copy()

        def mp(p):
            return (2 * c - p[0], p[1], wrap_angle(math.pi - p[2]))

        def mpoly(poly):
            out = poly.copy()
            out[:, 0] = 2 * c - out[:, 0]
            return out[::-1].copy()
    else:
        raise ValueError("axis must be 'horizontal' or 'vertical'")
    return Scenario(
        id=new_id or f"{sc.id}-mirror", name=name or f"{sc.name} (mirrored)", category=sc.category,
        description=f"Exact {axis} mirror of [{sc.id}] {sc.description}",
        grid=OccupancyGrid(data, g.resolution, g.origin), start=mp(sc.start), goal=mp(sc.goal),
        vehicle_params=dict(sc.vehicle_params), planner_overrides=dict(sc.planner_overrides),
        expectation=Expectation(**{f.name: getattr(sc.expectation, f.name) for f in fields(Expectation)}),
        obstacles=[Obstacle(mpoly(o.polygon), o.kind) for o in sc.obstacles],
        group=sc.group if group is None else group, tags=sc.tags)


def _with(sc: Scenario, **changes) -> Scenario:
    """Shallow copy of a scenario with some attributes replaced."""
    kw = {f.name: getattr(sc, f.name) for f in fields(Scenario)}
    kw.update(changes)
    return Scenario(**kw)


# ======================================================= reference motions
def f4_reference_arc(sc: Optional[Scenario] = None, ds: float = 0.05) -> np.ndarray:
    """Naive single left arc at maximum steering from F4's start to its goal
    (rear-axle point path). The point model finds it collision free, the
    footprint rectangle does not."""
    sc = sc or build_scenario("F4")
    v = sc.make_vehicle()
    r = v.min_turning_radius
    s = np.linspace(0.0, r * HALF_PI, int(math.ceil(r * HALF_PI / ds)) + 1)
    loc = arc_samples(1.0 / r, 1, s)
    x0, y0, th = sc.start
    c, si = math.cos(th), math.sin(th)
    out = np.empty_like(loc)
    out[:, 0] = x0 + c * loc[:, 0] - si * loc[:, 1]
    out[:, 1] = y0 + si * loc[:, 0] + c * loc[:, 1]
    out[:, 2] = np.array([wrap_angle(th + a) for a in loc[:, 2]])
    return out


F5_WALL_X = (20.0, 20.1)
F5_POLE = (28.3, 2.45, 0.08)   # thin round pole (cx, cy, r)
F5_ARC_START = (24.0, 3.0, 0.0)  # pose from which a curved primitive clips the pole mid-motion


def f5_reference_motion(ds: float = 0.05) -> np.ndarray:
    """1.4 m straight motion of the rear-axle point across F5's thin wall
    (both ends free for the point model, the middle is occupied)."""
    xs = np.arange(19.35, 20.75 + 1e-9, ds)
    return np.column_stack([xs, np.full_like(xs, 12.0), np.zeros_like(xs)])


def f5_reference_arc(vehicle: Optional[VehicleInfo] = None) -> np.ndarray:
    """Default-lattice forward, max-left-steer 'medium' primitive (1.19 m, 15 deg)
    from :data:`F5_ARC_START`, including its start pose. With the footprint
    rectangle (margin 0.1) its first and last poses are free of F5's pole but
    intermediate poses clip it with the front-right corner."""
    from motion_primitives import primitives_for
    v = vehicle or default_vehicle()
    ps = primitives_for(v)
    prim = [p for p in ps if p.direction == 1 and p.length_class == "medium"
            and abs(p.steering - v.max_steer_angle) < 1e-9][0]
    poses, _ = ps.transform_samples(*F5_ARC_START, np.arange(len(ps)) == prim.index)
    return np.vstack([np.array([F5_ARC_START]), poses])


# ============================================================ registry
_BUILDERS: Dict[str, Callable[[], Scenario]] = {}


def _register(sid: str):
    def deco(fn):
        _BUILDERS[sid] = fn
        return fn
    return deco


def _reg(sid: str, fn: Callable[[], Scenario]) -> None:
    _BUILDERS[sid] = fn


V = default_vehicle
D2R = math.radians


# ---------------------------------------------------------------- A basic
@_register("A1")
def _a1():
    return _make("A1", "Straight forward", "basic",
                 "Empty 30x14 m map, goal 15 m straight ahead with the same heading. "
                 "A pure forward straight line is optimal.",
                 open_area(30, 14), (5.0, 7.0, 0.0), (20.0, 7.0, 0.0),
                 expectation=Expectation(max_direction_changes=0, allowed_directions="forward"))


@_register("A2")
def _a2():
    return _make("A2", "Straight reverse", "basic",
                 "Empty 30x14 m map, goal 10 m straight behind the start with the same heading. "
                 "A pure reverse straight line is optimal (turning around costs far more).",
                 open_area(30, 14), (20.0, 7.0, 0.0), (10.0, 7.0, 0.0),
                 expectation=Expectation(max_direction_changes=0, allowed_directions="reverse"))


@_register("A3")
def _a3():
    return _make("A3", "Gentle turn (40 deg)", "basic",
                 "Empty 30x20 m map, goal 14 m ahead and 5 m to the left with heading +40 deg.",
                 open_area(30, 20), (4.0, 5.0, 0.0), (18.0, 10.0, D2R(40.0)),
                 expectation=Expectation(max_direction_changes=0, allowed_directions="forward"))


@_register("A4")
def _a4():
    return _make("A4", "90 deg turn", "basic",
                 "Empty 30x24 m map, 90 deg left turn: goal 9 m ahead, 9 m left, heading +90 deg "
                 "(radius twice R_min available).",
                 open_area(30, 24), (5.0, 4.0, 0.0), (14.0, 13.0, HALF_PI),
                 expectation=Expectation(max_direction_changes=0, allowed_directions="forward"))


@_register("A5")
def _a5():
    return _make("A5", "180 deg heading change", "basic",
                 "Empty 34x26 m map, U-turn: goal 12 m to the left, heading reversed. The lateral "
                 "offset exceeds 2*R_min so a forward-only U-turn exists.",
                 open_area(34, 26), (8.0, 7.0, 0.0), (8.0, 19.0, math.pi),
                 expectation=Expectation(max_direction_changes=2))


# -------------------------------------------------------- B perpendicular
def _perp(sid, name, desc, *, exp=None, tags=(), vehicle=None, **kw):
    v = vehicle or V()
    sc, s, g, _ = perpendicular_lot(v, **kw)
    return _make(sid, name, "perpendicular", desc, sc, s, g, v,
                 expectation=exp or Expectation(), tags=("perpendicular",) + tuple(tags))


B2_KW = dict(slot_width=2.5, aisle_width=6.5, neighbours="full", opposite="cars")


@_register("B1")
def _b1():
    return _perp("B1", "Wide slot, no neighbours",
                 "3.5 m slot with no parked neighbours, 7.0 m aisle, opposite row of cars. Start in "
                 "the aisle 7 m before the slot heading +x; forward past the slot then reverse in.",
                 slot_width=3.5, aisle_width=7.0, neighbours="none", opposite="cars",
                 exp=Expectation(min_direction_changes=1))


@_register("B2")
def _b2():
    return _perp("B2", "Normal slot, neighbours both sides",
                 "Commercial lot: 2.5 m slot pitch (3.15 m between neighbour cars, 0.525 m per side "
                 "beyond the margin), full row of parked cars, 6.5 m aisle, opposite row of cars.",
                 exp=Expectation(min_direction_changes=1), **B2_KW)


@_register("B3")
def _b3():
    return _perp("B3", "Narrow slot",
                 "2.35 m pitch (2.85 m between neighbours, 0.375 m per side beyond the margin), "
                 "full rows, 6.0 m aisle.",
                 slot_width=2.35, aisle_width=6.0, neighbours="full", opposite="cars",
                 exp=Expectation(min_direction_changes=1), tags=("narrow",))


@_register("B4")
def _b4():
    return _perp("B4", "Very narrow slot (near limit)",
                 "2.075 m pitch: 2.30 m between neighbours = vehicle width 1.90 + 2*0.1 margin + 0.2 m, "
                 "i.e. 0.1 m per side of real slack. 7.0 m aisle. Feasible by construction (straight "
                 "reverse into the slot), hard for the lattice.",
                 slot_width=2.075, aisle_width=7.0, neighbours="full", opposite="cars",
                 exp=Expectation(success=True, min_direction_changes=1,
                                 notes="feasible but near the limit; failure = planner weakness"),
                 tags=("narrow", "hard"))


@_register("B5")
def _b5():
    return _perp("B5", "Start offset left",
                 "B2 scene, start shifted 1.0 m to the left (away from the slot row).",
                 start_dy=1.0, exp=Expectation(min_direction_changes=1), **B2_KW)


@_register("B6")
def _b6():
    return _perp("B6", "Start offset right",
                 "B2 scene, start shifted 1.0 m to the right (towards the slot row, 0.2 m body "
                 "clearance to the neighbours' front bumpers).",
                 start_dy=-1.0, exp=Expectation(min_direction_changes=1), **B2_KW)


@_register("B7")
def _b7():
    return _perp("B7", "Start too close to slot",
                 "B2 scene, rear axle only 0.5 m before the slot centre line: there is no run-up "
                 "before the slot, the car is already alongside it.",
                 start_dx=-0.5, exp=Expectation(min_direction_changes=1), **B2_KW)


@_register("B8")
def _b8():
    return _perp("B8", "Far start",
                 "B2-type lot 45 m long, start 28 m before the slot.",
                 map_width=45.0, goal_x=34.0, start_dx=-28.0,
                 exp=Expectation(min_direction_changes=1), **B2_KW)


@_register("B9")
def _b9():
    return _perp("B9", "Initial yaw +15 deg",
                 "B2 scene, start heading +15 deg (pointing away from the slot row).",
                 start_yaw=D2R(15.0), exp=Expectation(min_direction_changes=1), **B2_KW)


@_register("B10")
def _b10():
    return _perp("B10", "Initial yaw -15 deg",
                 "B2 scene, start heading -15 deg (pointing towards the slot row).",
                 start_yaw=D2R(-15.0), exp=Expectation(min_direction_changes=1), **B2_KW)


# ------------------------------------------------------------ C parallel
def _par(sid, name, desc, *, exp=None, tags=(), vehicle=None, **kw):
    v = vehicle or V()
    sc, s, g, _ = parallel_lot(v, **kw)
    return _make(sid, name, "parallel", desc, sc, s, g, v,
                 expectation=exp or Expectation(), tags=("parallel",) + tuple(tags))


L0 = V().vehicle_length


@_register("C1")
def _c1():
    return _par("C1", "Wide parallel space",
                f"Space 2.0 x vehicle length ({2.0 * L0:.2f} m) between parked cars, 6.0 m road, start "
                "alongside the front car 1.0 m to its side. A single reverse S-manoeuvre suffices.",
                space_length=2.0 * L0, exp=Expectation(max_direction_changes=2))


@_register("C2")
def _c2():
    return _par("C2", "Normal parallel space",
                f"Space 1.5 x vehicle length ({1.5 * L0:.2f} m), 6.0 m road.", space_length=1.5 * L0,
                exp=Expectation(min_direction_changes=1))


@_register("C3")
def _c3():
    return _par("C3", "Tight parallel space",
                f"Space 1.3 x vehicle length ({1.3 * L0:.2f} m), 6.0 m road. With a 0.1 m continuous "
                "margin it needs ~7 gear changes, i.e. it is infeasible under the default product "
                "limits (<= 3 changes, >= 0.4 m manoeuvres); the expected answer is a fast rejection.",
                space_length=1.3 * L0,
                exp=Expectation(success=False, expected_status="no_path", max_time=20.0,
                                notes="too tight for <= 3 gear changes: must be rejected quickly"),
                tags=("narrow",))


@_register("C4")
def _c4():
    return _par("C4", "Near-minimum parallel space",
                "Space 5.40 m = vehicle length + 0.71 m (0.51 m longitudinal slack after the margin). "
                "Near the practical limit of multi-point parallel parking.",
                space_length=5.40,
                exp=Expectation(success=None, min_direction_changes=1,
                                notes="limit case; failure acceptable"), tags=("narrow", "hard"))


@_register("C5")
def _c5():
    return _par("C5", "Start very close to parked cars",
                f"C2 space ({1.5 * L0:.2f} m), start body side only 0.3 m from the parked cars "
                "(0.2 m beyond the margin).", space_length=1.5 * L0, start_gap=0.3,
                exp=Expectation(min_direction_changes=1), tags=("narrow",))


@_register("C6")
def _c6():
    base = _c2()
    return _with(mirror_scenario(base, "horizontal"), id="C6", name="Parallel, other side of the road",
                 description="C2 mirrored about the road centre line: the space is on the LEFT of the car "
                             "(one-way street / other side).", group="")


# ----------------------------------------------------------- D maneuvers
@_register("D1")
def _d1():
    # garage bay opening off a road bounded by walls
    sc = Scene(36.0, 16.0)
    gx, bay_w, road_y0, road_y1, bay_back = 20.0, 3.4, 1.0, 8.5, 14.5
    sc.box(0.0, 0.0, 36.0, road_y0, "wall")
    sc.box(0.0, road_y1, gx - bay_w / 2, 16.0, "wall")
    sc.box(gx + bay_w / 2, road_y1, 36.0, 16.0, "wall")
    sc.box(gx - bay_w / 2, bay_back, gx + bay_w / 2, 16.0, "wall")
    v = V()
    goal = (gx, bay_back - 0.3 - v.rear_overhang, -HALF_PI)
    start = (gx - 10.0, 0.5 * (road_y0 + road_y1), 0.0)
    return _make("D1", "One switch required", "maneuver",
                 "7.5 m road between walls with a 3.4 m wide, 6 m deep bay in the left wall. Goal: "
                 "reversed into the bay (nose to the road). Starting before the bay heading +x, reverse "
                 "only moves away and a forward loop does not fit, so >= 1 switch is required; one "
                 "forward-past + reverse-in manoeuvre suffices.",
                 sc, start, goal, v, expectation=Expectation(min_direction_changes=1),
                 tags=("maneuver",))


@_register("D2")
def _d2():
    v = V()
    sc, _, goal, info = perpendicular_lot(v, **B2_KW)
    # remove the parked car three slots to the left and put the ego car nose-in there
    sx = info.goal_x - 3 * info.slot_width
    sc.shapes = [s for s in sc.shapes
                 if not (s.kind == "vehicle" and abs(s.polygon[:, 0].mean() - sx) < 0.1
                         and s.polygon[:, 1].mean() < info.aisle_y0)]
    start = (sx, info.curb_top + 0.3 + v.wheel_base + v.front_overhang, -HALF_PI)
    return _make("D2", "Two switches required", "maneuver",
                 "B2 lot. The car is parked nose-in three slots to the left of the goal slot and must "
                 "reverse out, drive forward and reverse into the goal slot. The front is blocked by the "
                 "curb, and a reverse-only 180 deg turn does not fit in the 6.5 m aisle, so the sequence "
                 "R-F-R (>= 2 switches) is required.",
                 sc, start, goal, v, expectation=Expectation(min_direction_changes=2),
                 tags=("maneuver", "perpendicular"))


@_register("D3")
def _d3():
    return _make_d3("D3")


def _make_d3(sid, **extra):
    v = V()
    sc, s, g, _ = perpendicular_lot(v, slot_width=2.5, aisle_width=5.0, neighbours="full",
                                    opposite="cars")
    return _make(sid, "Narrow aisle perpendicular", "maneuver",
                 "Perpendicular lot with a 5.0 m aisle and 2.5 m pitch (full rows). The aisle is too "
                 "narrow to swing into the slot in one reverse move: multi-point manoeuvre expected.",
                 sc, s, g, v, expectation=Expectation(min_direction_changes=1),
                 tags=("maneuver", "narrow", "perpendicular"), **extra)


@_register("D4")
def _d4():
    v = V()
    sc, s, g, info = perpendicular_lot(v, slot_width=2.5, aisle_width=6.5, neighbours="full",
                                       opposite="cars", map_width=30.0, goal_x=24.0,
                                       dead_end_x=24.0 + 1.25 + 0.6, start_dx=-12.0)
    return _make("D4", "Dead end then reverse park", "maneuver",
                 "Aisle closed by a wall 0.6 m beyond the goal slot (the last slot of the row). The car "
                 "cannot pull past the slot before reversing, so it has to manoeuvre at the dead end.",
                 sc, s, g, v, expectation=Expectation(success=None, notes="benchmark only (hard)"),
                 tags=("maneuver", "perpendicular", "hard"))


@_register("D5")
def _d5():
    ix0, iy0, ix1, iy1 = 0.3, 0.3, 9.8, 7.3      # 9.5 x 7.0 m interior
    sc = Scene(10.1, 7.6).enclosure(ix0, iy0, ix1, iy1, 0.3)
    v = V()
    yc = 0.5 * (iy0 + iy1)
    start = (2.0, yc, 0.0)
    # same footprint position, heading reversed
    cx = start[0] + (v.wheel_base + v.front_overhang - v.rear_overhang) / 2
    goal = (2 * cx - start[0], yc, math.pi)
    return _make("D5", "Turn around inside a box", "maneuver",
                 "Closed 9.5 x 7.0 m box (walls all around). Turn the car by 180 deg on the spot "
                 "(same footprint position, reversed heading): a multi-point turn is required "
                 "(~8 gear changes: infeasible under the default <= 3 limit).",
                 sc, start, goal, v, expectation=Expectation(success=None, min_direction_changes=2,
                                                             notes="needs more than 3 gear changes"),
                 tags=("maneuver", "narrow", "hard"))


# ----------------------------------------------------------- E obstacles
S_E, G_E = (5.0, 10.0, 0.0), (34.0, 10.0, 0.0)


@_register("E1")
def _e1():
    sc = open_area(40, 20).box(18.5, 8.5, 21.5, 11.5)
    return _make("E1", "Single obstacle", "obstacle",
                 "3x3 m box on the straight line between start and goal (29 m apart).",
                 sc, S_E, G_E, expectation=Expectation(max_direction_changes=0), tags=("obstacle",))


@_register("E2")
def _e2():
    sc = open_area(40, 20).box(13.5, 9.0, 16.5, 14.0).box(23.5, 6.0, 26.5, 11.0)
    return _make("E2", "Two obstacles", "obstacle",
                 "Two 3x5 m boxes, staggered (one above, one below the direct line); the car can pass "
                 "either side of each.",
                 sc, S_E, G_E, expectation=Expectation(max_direction_changes=0), tags=("obstacle",))


@_register("E3")
def _e3():
    sc = open_area(40, 20).box(19.5, 0.0, 20.5, 8.8, "wall").box(19.5, 11.2, 20.5, 20.0, "wall")
    return _make("E3", "Narrow passage", "obstacle",
                 "Wall across the whole map with a single 2.4 m gap (vehicle 1.9 m + 2*0.1 m margin + "
                 "0.15 m per side), aligned with start and goal.",
                 sc, S_E, G_E, expectation=Expectation(max_direction_changes=0),
                 tags=("obstacle", "narrow"))


@_register("E4")
def _e4():
    sc = open_area(45, 20)
    sc.box(13.0, 0.0, 14.0, 7.5 - 1.3, "wall").box(13.0, 7.5 + 1.3, 14.0, 20.0, "wall")
    sc.box(30.0, 0.0, 31.0, 12.5 - 1.3, "wall").box(30.0, 12.5 + 1.3, 31.0, 20.0, "wall")
    return _make("E4", "Offset narrow passages", "obstacle",
                 "Two walls with 2.6 m gaps offset laterally by 5 m (gap 1 at y=7.5, gap 2 at y=12.5, "
                 "16 m apart): pass gap 1 straight, S-curve, pass gap 2 straight.",
                 sc, (4.0, 7.5, 0.0), (39.0, 12.5, 0.0),
                 expectation=Expectation(), tags=("obstacle", "narrow"))


@_register("E5")
def _e5():
    v = V()
    sc, s, g, info = perpendicular_lot(v, **B2_KW)
    gx = info.goal_x
    sc.box(gx + 1.15, info.aisle_y0 - 0.5, gx + 1.5, info.aisle_y0, "pillar")
    return _make("E5", "Pillar at slot entrance", "obstacle",
                 "B2 scene plus a 0.35x0.5 m pillar at the +x entrance corner of the goal slot, between "
                 "the slot and the neighbour car (0.1 m from the parked footprint incl. margin).",
                 sc, s, g, v, expectation=Expectation(min_direction_changes=1),
                 tags=("obstacle", "perpendicular"))


@_register("E6")
def _e6():
    v = V()
    sc, s, g, info = perpendicular_lot(v, **B2_KW)
    gx = info.goal_x
    y0 = info.curb_top + 0.3
    sc.box(gx + 1.15, y0, gx + 1.45, y0 + 0.6, "pillar")
    sc.box(gx - 1.45, y0, gx - 1.15, y0 + 0.6, "pillar")
    return _make("E6", "Pillars at rear corners", "obstacle",
                 "B2 scene plus 0.3x0.6 m pillars next to both rear corners of the goal slot (0.1 m "
                 "from the parked footprint incl. margin): the rear corners pass close while reversing.",
                 sc, s, g, v, expectation=Expectation(min_direction_changes=1),
                 tags=("obstacle", "perpendicular", "narrow"))


@_register("E7")
def _e7():
    sc = open_area(40, 16).box(0.0, 0.0, 40.0, 2.0, "wall").box(18.0, 2.0, 21.0, 5.0)
    return _make("E7", "Wall on one side", "obstacle",
                 "Long wall on the right (y<2). Goal 0.5 m from the wall (body clearance) behind a 3x3 m "
                 "block standing against the wall; the car must go around the block on the free side "
                 "and then approach the wall.",
                 sc, (5.0, 6.0, 0.0), (32.0, 3.45, 0.0), expectation=Expectation(),
                 tags=("obstacle",))


@_register("E8")
def _e8():
    h = 1.45        # half corridor width (vertical); narrowest perpendicular width ~2.81 m
    lower = [(0, 10 - h), (14, 10 - h), (22, 12 - h), (40, 12 - h)]
    upper = [(0, 10 + h), (14, 10 + h), (22, 12 + h), (40, 12 + h)]
    sc = open_area(40, 20)
    sc.polygon([(0, 0), (40, 0)] + [p for p in reversed(lower)], "wall")
    sc.polygon(upper + [(40, 20), (0, 20)], "wall")
    return _make("E8", "Narrow corridor (walls both sides)", "obstacle",
                 "2.9 m corridor (2.81 m in the 14 deg dog-leg) with walls on both sides: 0.35-0.4 m per "
                 "side beyond the margin.",
                 sc, (3.0, 10.0, 0.0), (36.0, 12.0, 0.0),
                 expectation=Expectation(max_direction_changes=0), tags=("obstacle", "narrow"))


@_register("E9")
def _e9():
    sc = open_area(40, 20).box(18.0, 6.0, 19.0, 15.0).box(11.0, 14.0, 18.0, 15.0)
    return _make("E9", "L-shaped obstacle", "obstacle",
                 "L-shaped obstacle made of two rectangles (vertical bar blocking the direct line, "
                 "horizontal bar extending towards the start at the top).",
                 sc, S_E, G_E, expectation=Expectation(), tags=("obstacle",))


@_register("E10")
def _e10():
    sc = open_area(40, 22)
    sc.box(22.0, 5.0, 23.0, 17.0).box(13.0, 4.0, 23.0, 5.0).box(13.0, 17.0, 23.0, 18.0)
    return _make("E10", "U-shaped trap", "obstacle",
                 "U-shaped obstacle (9 m deep, 12 m wide inside) opening towards the start, goal "
                 "directly behind it: a classic local minimum for goal-directed search.",
                 sc, (5.0, 11.0, 0.0), (32.0, 11.0, 0.0), expectation=Expectation(),
                 tags=("obstacle",))


# ---------------------------------------------------- F collision stress
@_register("F1")
def _f1():
    w, b = 4.1, 12.0         # corridor width, x of the vertical corridor's left wall
    y0 = 2.0
    width, height = b + w + 1.0, 20.0
    sc = Scene(width, height)
    sc.box(0.0, 0.0, width, y0, "wall")
    sc.box(b + w, y0, width, height, "wall")
    sc.box(0.0, y0 + w, b, height, "wall")
    return _make("F1", "Front outer corner close", "collision",
                 "Two 4.1 m corridors meeting at a right-angle L (left turn). During an R_min turn the "
                 "front outer corner sweeps out to 6.62 m from the turn centre (6.80 m with margin) while "
                 "the rear inner side stays 3.60 m from it. Verified with explicit straight-arc-straight "
                 "paths: a single R_min turn needs >= ~3.95 m, so at 4.1 m only a ~0.25 m window of turn "
                 "centres is collision-free and the front outer corner passes within a few cm (beyond "
                 "the margin) of the outer wall. Narrower corridors need reversing.",
                 sc, (4.0, y0 + w / 2, 0.0), (b + w / 2, 15.0, HALF_PI),
                 expectation=Expectation(), tags=("collision", "narrow"))


@_register("F2")
def _f2():
    gx, facade, back, half = 20.0, 10.0, 16.8, 1.3
    sc = Scene(40.0, 18.0)
    sc.box(0.0, facade, gx - half, facade + 0.3, "wall")
    sc.box(gx + half, facade, 40.0, facade + 0.3, "wall")
    sc.box(gx - half - 0.2, facade + 0.3, gx - half, back, "wall")
    sc.box(gx + half, facade + 0.3, gx + half + 0.2, back, "wall")
    sc.box(gx - half - 0.2, back, gx + half + 0.2, back + 0.2, "wall")
    v = V()
    goal = (gx, back - 0.3 - v.rear_overhang, -HALF_PI)
    return _make("F2", "Rear corners close (garage)", "collision",
                 "Reverse into a 2.6 m wide, 6.8 m deep garage in a building facade (0.25 m per side "
                 "beyond the margin). The rear corners pass the door jambs very closely.",
                 sc, (gx - 9.0, 5.0, 0.0), goal, v, expectation=Expectation(min_direction_changes=1),
                 tags=("collision", "narrow"))


@_register("F3")
def _f3():
    sc = open_area(40, 16).box(0.0, 0.0, 40.0, 8.55, "wall")
    sc.box(19.85, 10.8, 20.15, 11.1, "pillar")
    return _make("F3", "Side swipe", "collision",
                 "Straight run 5->35 m along a wall (y<8.55). A 0.3x0.3 m bollard at x=20 has its lower "
                 "face at y=10.8, i.e. 0.15 m inside the body side (0.25 m inside the margin) of the "
                 "straight line y=10: the straight line collides only with the body side (corners, "
                 "rear axle and both end poses are free). The free lane between wall and bollard is "
                 "2.25 m, so the car has to shift ~0.25 m right with 0.15 m total slack.",
                 sc, (5.0, 10.0, 0.0), (35.0, 10.0, 0.0),
                 expectation=Expectation(point_robot_should_succeed=True), tags=("collision", "narrow"))


@_register("F4")
def _f4():
    v = V()
    r = v.min_turning_radius
    start = (8.0, 4.0, 0.0)
    goal = (8.0 + r, 4.0 + r, HALF_PI)
    ctr = (8.0, 4.0 + r)
    psi, rho = D2R(60.0), 6.3
    px, py = ctr[0] + rho * math.sin(psi), ctr[1] - rho * math.cos(psi)
    sc = open_area(30, 20).circle(px, py, 0.3, "pillar")
    return _make("F4", "Rotation collision", "collision",
                 f"Goal lies exactly on the R_min left arc from the start (90 deg). A 0.3 m pillar at "
                 f"({px:.2f}, {py:.2f}) is 6.3 m from the turn centre: 1.75 m outside the rear-axle path "
                 "(point model: arc is collision-free, start/goal footprints free) but inside the "
                 "front-right corner's sweep (6.62 m) - the rectangle hits it around 25-45 deg into "
                 "the turn. See f4_reference_arc(). The rectangle planner must find another way.",
                 sc, start, goal, v,
                 expectation=Expectation(point_robot_should_succeed=True), tags=("collision",))


@_register("F5")
def _f5():
    sc = open_area(40, 20).box(F5_WALL_X[0], 5.5, F5_WALL_X[1], 20.0, "wall")
    sc.circle(*F5_POLE, kind="pillar")
    return _make("F5", "Thin obstacles (intermediate collision)", "collision",
                 "0.1 m thin wall across the direct path (x=20.0-20.1, y>5.5; detour below y=5.5) plus a "
                 f"0.16 m pole at ({F5_POLE[0]}, {F5_POLE[1]}) on the detour. (a) Rear-axle point: the 1.4 m "
                 "straight step 19.35->20.75 at y=12 has both endpoints free but occupied middle samples "
                 "(f5_reference_motion()). (b) Footprint: a straight step shorter than the vehicle can "
                 "never jump an obstacle (start and end footprints cover the swept area), but the "
                 "forward max-left 'medium' primitive (1.19 m, 15 deg) from (24, 3, 0) has free start and "
                 "end footprints while its front-right corner clips the pole mid-motion "
                 "(f5_reference_arc()). Only per-sample checking along primitives catches both.",
                 sc, (6.0, 12.0, 0.0), (34.0, 12.0, 0.0),
                 expectation=Expectation(), tags=("collision", "thin"))


# ---------------------------------------------------------- G unreachable
_G_EXP = dict(success=False, max_time=60.0)


@_register("G1")
def _g1():
    sc = open_area(40, 20).enclosure(26.0, 7.0, 36.0, 13.0, 0.3)
    return _make("G1", "Goal fully enclosed", "unreachable",
                 "Goal inside a closed 10x6 m room (0.3 m walls, no door).",
                 sc, S_E, (29.0, 10.0, 0.0), expectation=Expectation(**_G_EXP), tags=("unreachable",))


@_register("G2")
def _g2():
    sc = open_area(40, 20).box(19.5, 0.0, 20.5, 9.1, "wall").box(19.5, 10.9, 20.5, 20.0, "wall")
    return _make("G2", "Passage narrower than vehicle", "unreachable",
                 "Wall across the map with a single 1.8 m gap (vehicle is 1.9 m wide, 2.1 m with margin). "
                 "The only route. The gap is wider than the rear-axle clearance disc (~1.62 m) but the "
                 "coarse (0.2 m) 2D obstacle heuristic still closes it, so the planner currently rejects "
                 "it up front as 'unreachable'; a finer/less conservative heuristic would have to "
                 "exhaust the search or stop at its node / time limits instead.",
                 sc, S_E, G_E, expectation=Expectation(**_G_EXP), tags=("unreachable", "narrow"))


@_register("G3")
def _g3():
    sc = Scene(26.0, 4.0).enclosure(0.5, 0.7, 25.5, 3.3, 0.3)
    return _make("G3", "Turn-around impossible", "unreachable",
                 "Closed 25 x 2.6 m tube. Goal = same place with the heading reversed. In a 2.6 m wide "
                 "corridor the 4.89 x 2.10 m (with margin) footprint can yaw at most ~6 deg, so no "
                 "multi-point turn exists.",
                 sc, (5.0, 2.0, 0.0), (15.0, 2.0, math.pi), expectation=Expectation(**_G_EXP),
                 tags=("unreachable", "narrow"))


@_register("G4")
def _g4():
    sc = open_area(40, 20).box(18.5, 8.5, 21.5, 11.5)
    return _make("G4", "Start in collision", "unreachable",
                 "E1 map with the start placed so the footprint overlaps the 3x3 m box.",
                 sc, (17.0, 10.0, 0.0), G_E,
                 expectation=Expectation(success=False, expected_status="start_in_collision", max_time=5.0),
                 tags=("unreachable",))


@_register("G5")
def _g5():
    v = V()
    sc, s, g, info = perpendicular_lot(v, **B2_KW)
    goal = (info.goal_x + info.slot_width, g[1], g[2])
    return _make("G5", "Goal in collision", "unreachable",
                 "B2 lot with the goal set in the neighbouring slot, which is occupied by a parked car.",
                 sc, s, goal, v,
                 expectation=Expectation(success=False, expected_status="goal_in_collision", max_time=5.0),
                 tags=("unreachable",))


@_register("G6")
def _g6():
    v = V()
    sc, s, g, _ = perpendicular_lot(v, **B2_KW)
    params = vehicle_params_of(v)
    params.update(wheel_base=0.0, name="invalid_wheel_base")
    return _make("G6", "Invalid vehicle parameter", "unreachable",
                 "B2 lot with wheel_base = 0: VehicleInfo must raise InvalidVehicleParameterError.",
                 sc, s, g, vehicle_params=params,
                 expectation=Expectation(success=False, expect_exception=True), tags=("unreachable",))


@_register("G7")
def _g7():
    v = V()
    sc, s, g, _ = perpendicular_lot(v, **B2_KW)
    return _make("G7", "Goal outside the map", "unreachable",
                 "B2 lot with the goal 5 m beyond the right map edge: must be rejected as an invalid goal.",
                 sc, s, (sc.width + 5.0, g[1], g[2]), v,
                 expectation=Expectation(success=False, expected_status="goal_in_collision", max_time=5.0),
                 tags=("unreachable", "input"))


@_register("G8")
def _g8():
    sc = open_area(30, 20)
    p = (10.0, 10.0, 0.3)
    return _make("G8", "Start equals goal", "unreachable",
                 "Open area, goal identical to the start: must succeed immediately without moving.",
                 sc, p, p, expectation=Expectation(success=True, max_direction_changes=0, max_time=5.0),
                 tags=("input",))


@_register("G9")
def _g9():
    v = V()
    sc, s, g, _ = perpendicular_lot(v, **B2_KW)
    return _make("G9", "Unnormalised goal yaw", "unreachable",
                 "B2 lot with the goal yaw given as yaw + 4*pi (and the start yaw - 2*pi): must be handled "
                 "exactly like B2.",
                 sc, (s[0], s[1], s[2] - 2 * math.pi), (g[0], g[1], g[2] + 4 * math.pi), v,
                 expectation=Expectation(success=True, max_direction_changes=3), tags=("input",))


# ------------------------------------------------------- H vehicle variants
_H = [("H1", "Small car", small_car), ("H2", "Standard sedan", standard_sedan),
      ("H3", "Large SUV", large_suv), ("H4", "Long wheelbase", long_wheelbase),
      ("H5", "Sedan, reduced steering", reduced_steer_sedan)]


def _make_h(sid, name, preset):
    def build():
        v = preset()
        sc, s, g, _ = perpendicular_lot(v, **B2_KW)
        return _make(sid, name, "vehicle",
                     f"B2 scene with vehicle {v.summary()}; goal rear axle adapted to the rear overhang.",
                     sc, s, g, v, expectation=Expectation(min_direction_changes=1),
                     tags=("vehicle", "perpendicular"))
    return build


for _sid, _name, _preset in _H:
    _reg(_sid, _make_h(_sid, _name, _preset))


# ------------------------------------------------------ parameter sweeps
def _sweep(sid, name, category, group, base_fn, overrides, note):
    def build():
        base = base_fn()
        return _with(base, id=sid, name=name, category=category, group=group,
                     planner_overrides=dict(overrides),
                     description=f"[{base.id}] {base.description} -- sweep: {note}",
                     expectation=Expectation(success=None, notes=f"comparison group {group}"),
                     tags=tuple(base.tags) + ("sweep",))
    _reg(sid, build)


_B3 = lambda: _BUILDERS["B3"]()  # noqa: E731
_B2 = lambda: _BUILDERS["B2"]()  # noqa: E731
_D3 = lambda: _BUILDERS["D3"]()  # noqa: E731

for _r in (0.05, 0.10, 0.20):
    _sweep(f"I-xy{_r:.2f}", f"xy_resolution {_r:.2f} m", "resolution", "I-xy", _B3,
           {"xy_resolution": _r}, f"xy_resolution={_r}")
for _d in (2.5, 5.0, 10.0):
    _sweep(f"I-yaw{_d:g}", f"yaw_resolution {_d:g} deg", "resolution", "I-yaw", _B3,
           {"yaw_resolution": D2R(_d)}, f"yaw_resolution={_d} deg")
for _n in (3, 5, 7):
    _sweep(f"J-steer{_n}", f"n_steer {_n}", "primitive", "J-steer", _B2, {"n_steer": _n}, f"n_steer={_n}")
for _tag, _cls in (("s", ("short",)), ("sm", ("short", "medium")), ("sml", ("short", "medium", "long"))):
    _sweep(f"J-len-{_tag}", f"length classes {'/'.join(_cls)}", "primitive", "J-len", _B2,
           {"length_classes": _cls}, f"length_classes={_cls}")
for _p in (0.0, 5.0, 20.0):
    _sweep(f"K-switch{_p:g}", f"switch penalty {_p:g}", "cost", "K-switch", _D3,
           {"direction_switch_penalty": _p}, f"direction_switch_penalty={_p}")
for _p in (1.0, 1.5, 3.0):
    _sweep(f"K-reverse{_p:g}", f"reverse weight {_p:g}", "cost", "K-reverse", _D3,
           {"reverse_weight": _p}, f"reverse_weight={_p}")
for _p in (0.0, 0.3, 1.0):
    _sweep(f"K-steer{_p:g}", f"steer weight {_p:g}", "cost", "K-steer", _D3,
           {"steer_weight": _p}, f"steer_weight={_p}")
for _p in (0.0, 0.5, 2.0):
    _sweep(f"K-steerchange{_p:g}", f"steer change weight {_p:g}", "cost", "K-steerchange", _D3,
           {"steer_change_weight": _p}, f"steer_change_weight={_p}")
for _p in (0.0, 1.0, 5.0):
    _sweep(f"K-clearance{_p:g}", f"clearance weight {_p:g}", "cost", "K-clearance", _D3,
           {"clearance_weight": _p}, f"clearance_weight={_p}")
# footprint model: exact oriented rectangle vs conservative multi-circle approximation
for _base_id in ("B3", "E3"):
    for _m in ("rectangle", "circles"):
        _sweep(f"F-{_m}-{_base_id}", f"{_m} footprint on {_base_id}", "collision", f"F-method-{_base_id}",
               (lambda b=_base_id: _BUILDERS[b]()), {"collision_method": _m}, f"collision_method={_m}")
_B4 = lambda: _BUILDERS["B4"]()  # noqa: E731
_C3 = lambda: _BUILDERS["C3"]()  # noqa: E731
# switch penalty on a scene where maneuvers actually compete (very narrow slot)
for _p in (0.0, 5.0, 20.0):
    _sweep(f"K-switchB4-{_p:g}", f"switch penalty {_p:g} (narrow slot)", "cost", "K-switch-narrow", _B4,
           {"direction_switch_penalty": _p}, f"direction_switch_penalty={_p} on B4")
# safety margin vs feasibility in tight parallel parking
for _m in (0.0, 0.05, 0.10):
    _sweep(f"K-margin{_m:.2f}", f"safety margin {_m:.2f} m (tight parallel)", "cost", "K-margin", _C3,
           {"safety_margin": _m}, f"safety_margin={_m} on C3")
for _p in (0.05, 0.10, 0.20):
    _sweep(f"L-pos{_p:.2f}", f"position tolerance {_p:.2f} m", "tolerance", "L-pos", _B2,
           {"position_tolerance": _p}, f"position_tolerance={_p}")
for _d in (1.0, 3.0, 5.0):
    _sweep(f"L-yaw{_d:g}", f"yaw tolerance {_d:g} deg", "tolerance", "L-yaw", _B2,
           {"yaw_tolerance": D2R(_d)}, f"yaw_tolerance={_d} deg")
_sweep("L-noanalytic", "no analytic expansion, yaw tol 1 deg", "tolerance", "L-noanalytic", _B2,
       {"use_analytic_expansion": False, "yaw_tolerance": D2R(1.0)},
       "use_analytic_expansion=False, yaw_tolerance=1 deg (lattice must hit the goal on its own)")


# ------------------------------------------------------------ M symmetry
def _make_m(sid_base, name, group, base_fn, axis="horizontal"):
    def right():
        b = base_fn()
        return _with(b, id=f"{sid_base}-R", name=f"{name} (right)", category="symmetry", group=group,
                     description=f"Right-hand original of [{b.id}]: {b.description}",
                     tags=tuple(b.tags) + ("symmetry",))

    def left():
        return _with(mirror_scenario(right(), axis), id=f"{sid_base}-L", name=f"{name} (left)",
                     category="symmetry", group=group)
    _reg(f"{sid_base}-L", left)
    _reg(f"{sid_base}-R", right)


_make_m("M1", "Perpendicular", "M-perp", lambda: _BUILDERS["B5"]())
_make_m("M2", "Parallel", "M-par", lambda: _BUILDERS["C2"]())
_make_m("M3", "Obstacle avoidance", "M-obs", lambda: _BUILDERS["E2"]())
_make_m("M4", "Offset passages", "M-obs2", lambda: _BUILDERS["E4"]())


# ============================================================ N random
def random_scenario(index: int, seed: int = 42) -> Scenario:
    """Randomised parking case ``N-{index:03d}``; deterministic in (seed, index)."""
    from collision_checker import CollisionChecker

    rng = np.random.default_rng(seed + index)
    v = V()
    for attempt in range(100):
        sc_, start, goal, desc, keep_out = _random_layout(rng, v)
        # random obstacles (pillars / boxes) in the drivable area, away from goal slot & start
        n_obs = int(rng.integers(0, 4))
        placed = 0
        tries = 0
        while placed < n_obs and tries < 50:
            tries += 1
            x = float(rng.uniform(keep_out["xmin"], keep_out["xmax"]))
            y = float(rng.uniform(keep_out["ymin"], keep_out["ymax"]))
            if any(abs(x - zx) < zr and abs(y - zy) < zr for zx, zy, zr in keep_out["zones"]):
                continue
            if rng.random() < 0.5:
                sc_.circle(x, y, float(rng.uniform(0.2, 0.4)), "pillar")
            else:
                sc_.rect(x, y, float(rng.uniform(0.4, 1.2)), float(rng.uniform(0.4, 1.2)),
                         float(rng.uniform(0, math.pi)), "obstacle")
            placed += 1
        # rotate the whole lot by a random yaw and fit a compact map around it
        theta = float(rng.uniform(-math.pi, math.pi))
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        corners = np.array([[0, 0], [sc_.width, 0], [sc_.width, sc_.height], [0, sc_.height]]) @ rot.T
        lo = corners.min(axis=0)
        hi = corners.max(axis=0)
        trans = -lo + 0.5
        w, h = float(np.ceil(hi[0] - lo[0] + 1.0)), float(np.ceil(hi[1] - lo[1] + 1.0))
        scene = sc_.transformed(rot, trans, w, h)
        s = _transform_pose(start, rot, trans)
        g = _transform_pose(goal, rot, trans)
        grid = scene.rasterize()
        chk = CollisionChecker(grid, v, "rectangle", DEFAULT_MARGIN)
        if chk.check_pose(*s) or chk.check_pose(*g):
            continue
        # realism: after rasterising the rotated lot, the goal must keep at least
        # one grid cell of slack beyond the safety margin on every side
        # (rotated edges are inflated by up to ~1 cell by rasterisation)
        slack = CollisionChecker(grid, v, "rectangle", DEFAULT_MARGIN + grid.resolution)
        if slack.check_pose(*g):
            continue
        sid = f"N-{index:03d}"
        return Scenario(id=sid, name=f"Random {desc['kind']} #{index}", category="random",
                        description=(f"seed={seed} index={index} attempt={attempt}: {desc['text']}; "
                                     f"{placed} random obstacle(s); lot rotated by "
                                     f"{math.degrees(theta):.1f} deg."),
                        grid=grid, start=s, goal=g, vehicle_params=vehicle_params_of(v),
                        expectation=Expectation(success=None, notes="benchmark only"),
                        difficulty=desc["difficulty"], meta=desc["meta"],
                        obstacles=scene.obstacles(), group="N", tags=("random", desc["kind"]))
    raise RuntimeError(f"could not generate a collision-free random scenario for index {index}")


def _random_layout(rng: np.random.Generator, v: VehicleInfo):
    """One random (un-rotated) lot. Returns scene, start, goal, desc, keep_out."""
    kind = "perpendicular" if rng.random() < 0.5 else "parallel"
    left = bool(rng.random() < 0.5)
    if kind == "perpendicular":
        slot_w = float(rng.uniform(2.4, 3.0))
        aisle = float(rng.uniform(5.5, 8.0))
        nb = (int(rng.integers(0, 4)), int(rng.integers(0, 4)))
        if rng.random() < 0.4:
            nb = "full"
        forward_dir = rng.random() < 0.7
        dx = float(rng.uniform(-12.0, -3.0)) if forward_dir else float(rng.uniform(3.0, 12.0))
        yaw = (0.0 if forward_dir else math.pi) + D2R(float(rng.uniform(-15, 15)))
        opp = ["cars", "wall"][int(rng.integers(0, 2))]
        sc, s, g, info = perpendicular_lot(v, slot_width=slot_w, aisle_width=aisle, neighbours=nb,
                                           opposite=opp, map_width=34.0, goal_x=17.0, start_dx=dx,
                                           start_dy=float(rng.uniform(-0.5, 0.5)) * (aisle - 4.0) / 2,
                                           start_yaw=yaw, rng=rng, jitter=0.15,
                                           rear_gap=float(rng.uniform(0.35, 0.5)))
        difficulty = ("hard" if slot_w < 2.5 or aisle < 6.0 else
                      "easy" if slot_w >= 2.75 and aisle >= 7.0 else "normal")
        meta = dict(kind=kind, pitch=slot_w, aisle=aisle)
        text = (f"perpendicular, pitch {slot_w:.2f} m, aisle {aisle:.2f} m, neighbours {nb}, "
                f"opposite {opp}, start dx {dx:+.1f} m yaw {math.degrees(yaw):+.0f} deg")
        keep = dict(xmin=1.0, xmax=sc.width - 1.0, ymin=info.aisle_y0 + 0.5, ymax=info.aisle_y1 - 0.5,
                    zones=[(g[0], info.aisle_y0 + 2.0, 4.5), (s[0] + 1.3 * math.cos(yaw), s[1], 4.0)])
    else:
        L = float(rng.uniform(1.3, 1.8)) * v.vehicle_length
        road = float(rng.uniform(5.5, 7.0))
        gap = float(rng.uniform(0.6, 1.5))
        # realistic final distance to the curb (bare body): 0.30-0.45 m
        curb_gap = float(rng.uniform(0.30, 0.45))
        sc, s, g, info = parallel_lot(v, space_length=L, road_width=road, map_width=36.0, space_x=14.0,
                                      n_cars=(int(rng.integers(1, 3)), int(rng.integers(1, 3))),
                                      curb_gap=curb_gap,
                                      start_dx=float(rng.uniform(0.0, 3.0)), start_gap=gap,
                                      start_yaw=D2R(float(rng.uniform(-10, 10))), rng=rng, jitter=0.15)
        ratio = L / v.vehicle_length
        difficulty = "hard" if ratio < 1.45 else "easy" if ratio >= 1.65 and road >= 6.0 else "normal"
        meta = dict(kind=kind, ratio=ratio, road=road)
        text = (f"parallel, space {L:.2f} m ({L / v.vehicle_length:.2f} x L), road {road:.2f} m, "
                f"gap {gap:.2f} m, curb gap {curb_gap:.2f} m")
        keep = dict(xmin=1.0, xmax=sc.width - 1.0, ymin=info["lane_edge"] + 2.8,
                    ymax=info["road_y1"] - 0.5,
                    zones=[(g[0] + 1.3, g[1] + 2.0, 6.0), (s[0] + 1.3, s[1], 4.5)])
    # enclose the lot so the rotated map corners are not a free shortcut
    sc.enclosure(0.3, 0.3, sc.width - 0.3, sc.height - 0.3, 0.3)
    if left:
        # mirror the lot (slot on the left of the driving direction)
        H = sc.height
        m = np.array([[1.0, 0.0], [0.0, -1.0]])
        sc = sc.transformed(m, (0.0, H), sc.width, H)
        s = _transform_pose(s, m, (0.0, H))
        g = _transform_pose(g, m, (0.0, H))
        keep = dict(keep, ymin=H - keep["ymax"], ymax=H - keep["ymin"],
                    zones=[(zx, H - zy, zr) for zx, zy, zr in keep["zones"]])
        text += ", mirrored (slot on the left)"
    return sc, s, g, dict(kind=kind, text=text, difficulty=difficulty, meta=meta), keep


def random_scenarios(n: int, seed: int = 42) -> List[Scenario]:
    return [random_scenario(i, seed) for i in range(n)]


# ============================================================ difficulty
_EASY = {"A1", "A2", "A3", "A4", "A5", "B1", "C1", "E1", "E2", "E7", "E8"}
_HARD = {"B4", "B8", "C3", "C4", "D4", "D5", "E10", "K-margin0.00", "K-margin0.05", "K-margin0.10",
         "K-switchB4-0", "K-switchB4-5", "K-switchB4-20"}


def difficulty_of(sc: Scenario) -> str:
    """Benchmark difficulty. Random cases get it from their geometry (slot pitch /
    aisle width, parallel space length); fixed cases from the table above.
    Invalid-input / unreachable cases are not part of the benchmark ("n/a")."""
    if sc.difficulty:
        return sc.difficulty
    if sc.category == "unreachable":
        return "n/a"
    if sc.id in _EASY:
        return "easy"
    if sc.id in _HARD:
        return "hard"
    return "normal"


# ================================================================= product spec
# Allowed gear changes per scene (None = no spec). "reject" = a fast
# "slot too tight" answer is also acceptable. F1: the 0-change corner turn has
# only a few cm of slack; one reverse correction is acceptable there.
_SPEC = {
    "A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0,
    "B1": 1, "B2": 1, "B3": 1, "B4": 3, "B5": 1, "B6": 1, "B7": 1, "B8": 1, "B9": 1, "B10": 1,
    "C1": 1, "C2": 2, "C3": (3, "reject"), "C4": (3, "reject"), "C5": 2, "C6": 2,
    "D1": 1, "D2": 2, "D3": 2, "D4": (3, "reject"), "D5": (3, "reject"),
    "E1": 0, "E2": 0, "E3": 0, "E4": 0, "E5": 1, "E6": 1, "E7": 0, "E8": 0, "E9": 0, "E10": 0,
    "F1": 1, "F2": 1, "F3": 0, "F4": 0, "F5": 0,
    "H1": 1, "H2": 1, "H3": 1, "H4": 1, "H5": 1,
    "M1-L": 1, "M1-R": 1, "M2-L": 2, "M2-R": 2, "M3-L": 0, "M3-R": 0, "M4-L": 0, "M4-R": 0,
}


def spec_of(sc: Scenario):
    """(max gear changes, rejection acceptable) of the product specification:

    * perpendicular: 1 (pitch >= 2.5 m and aisle >= 6 m), else 3 (reverse entry
      from a forward start: 2 gear changes are never enough, only 1 or 3)
    * parallel: 1 (space >= 1.6 L), 2 (>= 1.45 L), else 3 or a fast rejection
    * open space / obstacle avoidance: no gear change
    Parameter sweeps and invalid-input tests have no spec (None)."""
    if sc.category == "random":
        m = sc.meta
        if m.get("kind") == "perpendicular":
            return (1 if m["pitch"] >= 2.5 and m["aisle"] >= 6.0 else 3), False
        r = m.get("ratio", 0.0)
        return (1, False) if r >= 1.6 else (2, False) if r >= 1.45 else (3, True)
    v = _SPEC.get(sc.id)
    if v is None:
        return None, False
    return (v[0], True) if isinstance(v, tuple) else (v, False)


def _finish(sc: Scenario) -> Scenario:
    sc.difficulty = difficulty_of(sc)
    sc.spec_max_switches, sc.spec_reject_ok = spec_of(sc)
    return sc


# ================================================================ public
def scenario_ids() -> List[str]:
    """Ids of all fixed scenarios, in canonical order."""
    return list(_BUILDERS.keys())


def build_scenario(scenario_id: str, seed: int = 42) -> Scenario:
    """Deterministically (re)build one scenario by id."""
    if scenario_id.startswith("N-"):
        return _finish(random_scenario(int(scenario_id[2:]), seed))
    try:
        return _finish(_BUILDERS[scenario_id]())
    except KeyError:
        raise KeyError(f"unknown scenario id {scenario_id!r}") from None


def all_scenarios(include_random: int = 0, seed: int = 42) -> List[Scenario]:
    out = [_finish(fn()) for fn in _BUILDERS.values()]
    if include_random:
        out += [_finish(sc) for sc in random_scenarios(include_random, seed)]
    return out


def _resolve_category(category: str):
    c = category.strip()
    if c.lower() == "narrow":
        return ("tag", "narrow")
    if c in CATEGORIES:
        return ("cat", c)
    if c.upper() in _LETTER_TO_CATEGORY and len(c) == 1:
        return ("cat", _LETTER_TO_CATEGORY[c.upper()])
    for k, (_, disp) in CATEGORIES.items():
        if c.lower() in (k.lower(), disp.lower()):
            return ("cat", k)
    raise KeyError(f"unknown category {category!r}; use one of {list(CATEGORIES)}, a letter or 'narrow'")


def get_scenarios(category: Optional[str] = None, ids: Optional[List[str]] = None, random_n: int = 0,
                  seed: int = 42) -> List[Scenario]:
    """Select scenarios by category (key, letter, display name or 'narrow') and/or ids.

    Random cases are included when ``random_n > 0``; selecting the random
    category without ``random_n`` yields 20 random cases.
    """
    if ids:
        out = [build_scenario(i, seed) for i in ids]
    else:
        n_rand = random_n
        if category is not None and _resolve_category(category) == ("cat", "random") and n_rand == 0:
            n_rand = 20
        out = all_scenarios(n_rand, seed)
    if category is not None:
        mode, val = _resolve_category(category)
        out = [s for s in out if (s.category == val if mode == "cat" else val in s.tags)]
    return out


__all__ = [
    "CATEGORIES", "Expectation", "Obstacle", "Scenario", "Scene", "LotInfo", "open_area",
    "perpendicular_lot", "parallel_lot", "mirror_scenario", "vehicle_params_of", "all_scenarios",
    "get_scenarios", "build_scenario", "random_scenarios", "random_scenario", "scenario_ids",
    "f4_reference_arc", "f5_reference_motion",
]
