"""Safety guarantees of the search.

* the safety margin holds for the continuous motion (not only at samples),
* the endpoint margin relaxation only lowers the margin near the endpoints,
  never below the endpoint's own clearance, and never accepts a colliding pose.
"""
import math

import numpy as np
import pytest

from metrics import IndependentValidator
from occupancy_grid import OccupancyGrid
from state_lattice import PlannerConfig, PlanStatus, StateLatticePlanner, inter_sample_bound
from vehicle import default_vehicle


def test_inter_sample_bound_matches_brute_force():
    """The bound must dominate the true deviation of any footprint point between
    two samples along a full-lock arc (checked numerically)."""
    v = default_vehicle()
    ds, m = 0.05, 0.1
    bound = inter_sample_bound(v, m, ds)
    k = v.max_curvature
    corners = v.footprint_local(m)
    worst = 0.0
    for sgn in (1, -1):
        kk = sgn * k
        for x, y in corners:
            # body point trajectory for rear-axle arc length s in [0, ds]
            s = np.linspace(0, ds, 201)
            th = kk * s
            px = np.sin(th) / kk + x * np.cos(th) - y * np.sin(th)
            py = (1 - np.cos(th)) / kk + x * np.sin(th) + y * np.cos(th)
            d0 = np.hypot(px - px[0], py - py[0])
            d1 = np.hypot(px - px[-1], py - py[-1])
            worst = max(worst, float(np.max(np.minimum(d0, d1))))
    assert worst <= bound + 1e-9
    assert bound < 1.5 * worst  # and it is not wildly conservative


@pytest.mark.parametrize("sid", ["B4", "C2", "E6", "F1", "F3"])
def test_trajectories_keep_the_margin_continuously(sid):
    """Tight scenes: the returned path keeps ``safety_margin`` for the continuous
    motion (independent shapely check, densified to 1 cm)."""
    from scenarios import build_scenario
    sc = build_scenario(sid)
    v = sc.make_vehicle()
    from parking_planner import BidirectionalPlanner
    cfg = sc.make_config(PlannerConfig(max_planning_time=60.0, endpoint_relaxation=False))
    res = BidirectionalPlanner(v, cfg).plan(sc.grid, sc.start, sc.goal)
    assert res.success, res.message
    val = IndependentValidator(sc.grid, v, margin=cfg.safety_margin - 1e-3)
    assert not val.colliding_indices(res.trajectory.poses(), densify_step=0.01)


def _pillar_start_map():
    g = OccupancyGrid.empty(30, 16, 0.1)
    g.add_box(8.0, 9.05, 9.0, 10.0)   # pillar 0.05 m beside the start footprint's left side
    return g


def test_start_closer_than_margin_is_planned_with_relaxation():
    v = default_vehicle()
    g = _pillar_start_map()
    start = (6.0, 8.0, 0.0)  # left side at y = 8 + 0.95 = 8.95 -> 0.10 m gap to the pillar... minus cells
    goal = (22.0, 8.0, 0.0)
    strict = StateLatticePlanner(v, PlannerConfig(endpoint_relaxation=False, max_planning_time=5)).plan(g, start, goal)
    assert strict.status == PlanStatus.START_IN_COLLISION
    relaxed = StateLatticePlanner(v, PlannerConfig(max_planning_time=5)).plan(g, start, goal)
    assert relaxed.success, relaxed.message
    assert "start margin relaxed" in relaxed.message
    # never an actual (bare footprint) collision
    assert IndependentValidator(g, v, margin=0.0).is_collision_free(relaxed.trajectory.poses())


def test_relaxation_never_accepts_a_bare_collision():
    v = default_vehicle()
    g = OccupancyGrid.empty(30, 16, 0.1)
    g.add_box(8.0, 8.5, 9.0, 10.0)   # pillar overlapping the start footprint
    res = StateLatticePlanner(v, PlannerConfig(max_planning_time=5)).plan(g, (6.0, 8.0, 0.0), (22.0, 8.0, 0.0))
    assert res.status == PlanStatus.START_IN_COLLISION


def test_relaxation_is_local_to_the_endpoint():
    """Far from the start the nominal margin applies again."""
    v = default_vehicle()
    g = _pillar_start_map()
    # second pillar near the middle of the straight path, 0.06 m from the body side
    g.add_box(15.0, 9.01, 16.0, 10.0)
    cfg = PlannerConfig(max_planning_time=10)
    res = StateLatticePlanner(v, cfg).plan(g, (6.0, 8.0, 0.0), (22.0, 8.0, 0.0))
    assert res.success
    poses = res.trajectory.poses()
    far = np.hypot(poses[:, 0] - 6.0, poses[:, 1] - 8.0) > v.vehicle_length + 0.5
    val = IndependentValidator(g, v, margin=cfg.safety_margin - 1e-3)
    assert not val.colliding_indices(poses[far], densify_step=0.02)


def test_relaxation_only_applies_to_the_constraining_obstacle():
    """Regression (independent review): the start is 3 cm from a wall, so its zone
    is relaxed. A pose 4.5 m away that is 6 cm from a *different* obstacle must
    still be rejected (that obstacle keeps the full margin)."""
    v = default_vehicle()
    res = 0.02
    g = OccupancyGrid(np.zeros((1500, 1500), np.uint8), res, (0.0, 0.0))
    start = (5.0, 10.02, 0.0)
    g.data[550:575, 150:450] = 1                       # wall 3 cm left of the start footprint
    yaw = math.radians(-50)
    p = (start[0] + 4.3, start[1] - 1.3, yaw)
    fl = v.footprint_world(*p)[1]
    cx = fl[0] + 0.065 * math.cos(yaw + math.pi / 4)
    cy = fl[1] + 0.065 * math.sin(yaw + math.pi / 4)
    jx, jy = int(math.floor(cx / res)), int(math.floor(cy / res))
    g.data[jy:jy + 5, jx:jx + 5] = 1                   # second obstacle ~6 cm from the pose
    pl = StateLatticePlanner(v, PlannerConfig(min_relaxed_margin=0.0))  # allow relaxing this start
    zoned, notes = pl.make_zoned_checker(g, (("start", start),))
    assert notes, "the start should be relaxed"
    assert not zoned.check_pose(*start)
    assert pl.make_checker(g).check_pose(*p)
    assert zoned.check_pose(*p), "relaxation leaked to an obstacle that does not constrain the start"
