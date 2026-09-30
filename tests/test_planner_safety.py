"""Safety / robustness / output-validity tests for the state-lattice planner."""
import math
import time

import numpy as np
import pytest
import shapely

import state_lattice
from collision_checker import CollisionChecker
from occupancy_grid import OccupancyGrid
from state_lattice import PlannerConfig, PlanStatus, StateLatticePlanner
from vehicle import default_vehicle, wrap_angle

FAST = dict(max_planning_time=5.0, max_expanded_nodes=20_000)


def open_map():
    g = OccupancyGrid.empty(20.0, 15.0, 0.1)
    g.add_border(0.2)
    return g


def wall_map():
    """Wall with a narrow gap at the far end: hard (search does not finish in seconds)."""
    g = open_map()
    g.add_box(9.5, 0.0, 10.5, 12.0)
    return g


def obstacle_map():
    """Wall hanging from the top: the vehicle must U-turn below it."""
    g = open_map()
    g.add_box(8.0, 6.0, 10.0, 15.0)
    return g


OBS_START, OBS_GOAL = (4.0, 10.0, -math.pi / 2), (14.0, 10.0, math.pi / 2)


def enclosed_goal_map():
    g = open_map()
    g.add_box(11.0, 3.0, 19.0, 3.3)
    g.add_box(11.0, 11.7, 19.0, 12.0)
    g.add_box(11.0, 3.0, 11.3, 12.0)
    g.add_box(18.7, 3.0, 19.0, 12.0)
    return g


@pytest.fixture(scope="module")
def planner():
    p = StateLatticePlanner(default_vehicle(), PlannerConfig(**FAST))
    # warm-up (JIT kernels, heuristic tables) so timing assertions measure the planner itself
    p.plan(open_map(), (4.0, 7.5, 0.0), (6.0, 7.5, 0.0))
    return p


def _timed(planner, grid, start, goal):
    t0 = time.perf_counter()
    r = planner.plan(grid, start, goal)
    return r, time.perf_counter() - t0


# ------------------------------------------------------------ fail fast
def test_start_in_collision(planner):
    g = open_map()
    g.add_box(3.0, 7.0, 4.0, 8.0)
    r, dt = _timed(planner, g, (3.5, 7.5, 0.0), (14.0, 7.5, 0.0))
    assert r.status == PlanStatus.START_IN_COLLISION
    assert dt < 1.0 and r.expanded_nodes == 0 and r.trajectory is None
    # start partly outside the map
    r, dt = _timed(planner, open_map(), (0.3, 7.5, 0.0), (14.0, 7.5, 0.0))
    assert r.status == PlanStatus.START_IN_COLLISION and dt < 1.0


def test_goal_in_collision(planner):
    g = open_map()
    g.add_box(14.0, 7.0, 15.0, 8.0)
    r, dt = _timed(planner, g, (4.0, 7.5, 0.0), (14.0, 7.5, 0.0))
    assert r.status == PlanStatus.GOAL_IN_COLLISION
    assert dt < 1.0 and r.trajectory is None
    r = planner.plan(open_map(), (4.0, 7.5, 0.0), (19.0, 7.5, 0.0))  # goal footprint outside map
    assert r.status == PlanStatus.GOAL_IN_COLLISION


@pytest.mark.parametrize("start, goal", [
    ((float("nan"), 7.5, 0.0), (14.0, 7.5, 0.0)),
    ((4.0, 7.5, 0.0), (14.0, float("nan"), 0.0)),
    ((4.0, 7.5, float("inf")), (14.0, 7.5, 0.0)),
    ((4.0, 7.5), (14.0, 7.5, 0.0)),
    ((4.0, 7.5, 0.0, 1.0), (14.0, 7.5, 0.0)),
    (None, (14.0, 7.5, 0.0)),
    (("a", 1, 2), (14.0, 7.5, 0.0)),
])
def test_invalid_input(planner, start, goal):
    r = planner.plan(open_map(), start, goal)
    assert r.status == PlanStatus.INVALID_INPUT
    assert not r.success and r.trajectory is None


def test_invalid_config_rejected():
    for kw in (dict(heuristic_weight=0.5), dict(reverse_weight=0.9), dict(max_expanded_nodes=0),
               dict(collision_method="blob"), dict(heuristic="nope"), dict(n_steer=4)):
        with pytest.raises(ValueError):
            StateLatticePlanner(default_vehicle(), PlannerConfig(**kw))


def test_enclosed_goal_fails_quickly(planner):
    r, dt = _timed(planner, enclosed_goal_map(), (4.0, 7.5, 0.0), (13.5, 7.5, 0.0))
    assert r.status in (PlanStatus.UNREACHABLE, PlanStatus.NO_PATH)
    assert r.status == PlanStatus.UNREACHABLE, "2D obstacle heuristic should prove unreachability"
    assert dt < 2.0 and r.trajectory is None


def test_enclosed_goal_without_obstacle_heuristic():
    """Without the 2D proof the search must still terminate within its limits."""
    cfg = PlannerConfig(heuristic="reeds_shepp", max_expanded_nodes=3000, max_planning_time=2.0)
    p = StateLatticePlanner(default_vehicle(), cfg)
    r, dt = _timed(p, enclosed_goal_map(), (4.0, 7.5, 0.0), (13.5, 7.5, 0.0))
    assert r.status in (PlanStatus.NO_PATH, PlanStatus.MAX_EXPANSIONS, PlanStatus.TIMEOUT)
    assert r.expanded_nodes <= 3000 and dt < 4.0


# ------------------------------------------------------------ limits
@pytest.mark.parametrize("limit", [1, 50, 400])
def test_max_expanded_nodes_respected(limit):
    p = StateLatticePlanner(default_vehicle(), PlannerConfig(max_expanded_nodes=limit, max_planning_time=10.0))
    r = p.plan(wall_map(), (4.0, 5.0, 0.0), (15.0, 5.0, 0.0))
    assert r.status == PlanStatus.MAX_EXPANSIONS
    assert r.expanded_nodes <= limit
    assert r.trajectory is None


@pytest.mark.parametrize("limit", [0.001, 0.3])
def test_max_planning_time_respected(limit):
    p = StateLatticePlanner(default_vehicle(), PlannerConfig(max_planning_time=limit, max_expanded_nodes=10 ** 7))
    r, dt = _timed(p, wall_map(), (4.0, 5.0, 0.0), (15.0, 5.0, 0.0))
    assert r.status == PlanStatus.TIMEOUT
    # heuristic construction happens before the first check; allow a small overshoot
    assert dt < limit + 1.0
    assert r.planning_time < limit + 1.0


# ------------------------------------------------------------ robustness
def test_planner_never_raises_build_heuristic(planner, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(state_lattice, "build_heuristic", boom)
    r = planner.plan(open_map(), (4.0, 7.5, 0.0), (14.0, 7.5, 0.0))
    assert r.status == PlanStatus.ERROR
    assert "injected failure" in r.message


def test_planner_never_raises_collision_checker(monkeypatch):
    p = StateLatticePlanner(default_vehicle(), PlannerConfig(**FAST))

    class Broken:
        def __init__(self, *a, **k):
            raise MemoryError("no memory")
    monkeypatch.setattr(state_lattice, "CollisionChecker", Broken)
    r = p.plan(open_map(), (4.0, 7.5, 0.0), (14.0, 7.5, 0.0))
    assert r.status == PlanStatus.ERROR


def test_planner_never_raises_in_search(monkeypatch):
    p = StateLatticePlanner(default_vehicle(), PlannerConfig(use_analytic_expansion=False, **FAST))

    def boom(*a, **k):
        raise ValueError("expansion failure")
    monkeypatch.setattr(p.primitives, "transform_samples", boom)
    r = p.plan(open_map(), (4.0, 7.5, 0.0), (14.0, 9.0, 0.5))
    assert r.status == PlanStatus.ERROR and "expansion failure" in r.message


def test_planner_handles_bad_grid(planner):
    r = planner.plan(None, (4.0, 7.5, 0.0), (14.0, 7.5, 0.0))
    assert r.status == PlanStatus.ERROR


@pytest.mark.parametrize("analytic", [True, False])
def test_determinism(analytic):
    cfg = PlannerConfig(use_analytic_expansion=analytic, **FAST)
    g = obstacle_map()
    res = [StateLatticePlanner(default_vehicle(), cfg).plan(g, OBS_START, OBS_GOAL)
           for _ in range(2)]
    a, b = res
    assert a.status == b.status == PlanStatus.SUCCESS
    assert a.expanded_nodes == b.expanded_nodes
    assert a.generated_nodes == b.generated_nodes
    assert a.collision_checks == b.collision_checks
    assert a.cost == b.cost
    np.testing.assert_array_equal(a.trajectory.poses(), b.trajectory.poses())
    np.testing.assert_array_equal(a.trajectory.steerings(), b.trajectory.steerings())
    np.testing.assert_array_equal(a.trajectory.directions(), b.trajectory.directions())


# ------------------------------------------------------------ output validity
CASES = {
    "open_analytic": (dict(), open_map, (4.0, 7.5, 0.0), (14.0, 9.0, 0.5)),
    "open_lattice": (dict(use_analytic_expansion=False), open_map, (4.0, 7.5, 0.0), (14.0, 9.0, 0.5)),
    "reverse": (dict(), open_map, (10.0, 7.5, 0.0), (5.0, 6.5, 0.0)),
    "around_wall": (dict(), obstacle_map, OBS_START, OBS_GOAL),
    "around_wall_lattice": (dict(use_analytic_expansion=False), obstacle_map, OBS_START, OBS_GOAL),
    "circles": (dict(collision_method="circles"), obstacle_map, OBS_START, OBS_GOAL),
}


@pytest.fixture(scope="module", params=list(CASES))
def solved(request):
    kw, mk, start, goal = CASES[request.param]
    cfg = PlannerConfig(**{**FAST, **kw})
    v = default_vehicle()
    g = mk()
    t0 = time.perf_counter()
    r = StateLatticePlanner(v, cfg).plan(g, start, goal)
    return dict(name=request.param, result=r, cfg=cfg, vehicle=v, grid=g, start=start, goal=goal,
                dt=time.perf_counter() - t0)


def test_plan_succeeds(solved):
    r = solved["result"]
    assert r.status == PlanStatus.SUCCESS, f"{solved['name']}: {r.status} {r.message[:200]}"
    assert solved["dt"] < 5.0
    assert r.trajectory is not None and len(r.trajectory) >= 2
    assert math.isfinite(r.cost) and r.cost > 0
    np.testing.assert_allclose(r.trajectory.poses()[0], solved["start"], atol=1e-9)


def test_trajectory_collision_free(solved):
    r, v, g, cfg = solved["result"], solved["vehicle"], solved["grid"], solved["cfg"]
    poses = r.trajectory.poses()
    cc = CollisionChecker(g, v, "rectangle", cfg.safety_margin if cfg.collision_method == "rectangle" else 0.0, 4)
    assert not cc.check_poses(poses).any()
    # independent check of the bare footprint with shapely
    iy, ix = np.nonzero(g.data)
    res = g.resolution
    cells = shapely.box(ix * res, iy * res, (ix + 1) * res, (iy + 1) * res)
    tree = shapely.STRtree(cells)
    for p in poses:
        poly = shapely.Polygon(v.footprint_world(*p))
        for j in tree.query(poly, predicate="intersects"):
            assert cells[j].intersection(poly).area < 1e-9, f"footprint overlaps obstacle at pose {p}"


def test_trajectory_steering_within_limits(solved):
    r, v = solved["result"], solved["vehicle"]
    st = r.trajectory.steerings()
    assert np.all(np.abs(st) <= v.max_steer_angle + 1e-9)
    for s in r.segments:
        assert abs(s.steering) <= v.max_steer_angle + 1e-9


def test_trajectory_curvature_consistent(solved):
    r, v = solved["result"], solved["vehicle"]
    tr = r.trajectory
    np.testing.assert_allclose(tr.curvatures(), np.tan(tr.steerings()) / v.wheel_base, atol=1e-12)
    assert np.all(np.abs(tr.curvatures()) <= v.max_curvature + 1e-9)
    # geometric curvature of consecutive points matches the commanded curvature
    p = tr.poses()
    d = tr.directions()
    ds = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    dyaw = np.array([wrap_angle(a) for a in np.diff(p[:, 2])])
    ok = ds > 1e-6
    geo = dyaw[ok] / (d[1:][ok] * ds[ok])
    np.testing.assert_allclose(geo, tr.curvatures()[1:][ok], rtol=1e-3, atol=1e-6)
    # the motion direction agrees with the heading (forward: displacement along heading)
    along = np.cos(p[:-1, 2]) * np.diff(p[:, 0]) + np.sin(p[:-1, 2]) * np.diff(p[:, 1])
    assert np.all(np.sign(along[ok]) == d[1:][ok])
    # kinematic continuity: no jumps
    assert ds.max() <= solved["cfg"].sample_ds + 1e-6
    assert np.abs(dyaw).max() <= v.max_curvature * solved["cfg"].sample_ds + 1e-6


def test_goal_error_within_tolerance(solved):
    r, cfg, goal = solved["result"], solved["cfg"], solved["goal"]
    end = r.trajectory.poses()[-1]
    assert math.hypot(end[0] - goal[0], end[1] - goal[1]) <= cfg.position_tolerance + 1e-9
    assert abs(wrap_angle(end[2] - goal[2])) <= cfg.yaw_tolerance + 1e-9
    if r.analytic_success:
        assert math.hypot(end[0] - goal[0], end[1] - goal[1]) < 1e-3


def test_arc_length_monotonic(solved):
    tr = solved["result"].trajectory
    s = np.array([pt.arc_length for pt in tr])
    assert s[0] == 0.0
    assert np.all(np.diff(s) >= 0)
    p = tr.poses()
    assert s[-1] == pytest.approx(np.hypot(*np.diff(p[:, :2], axis=0).T).sum())
    fwd, rev = tr.directional_lengths()
    assert fwd + rev == pytest.approx(tr.length)
    # path length consistent with the segment breakdown (chords <= arcs)
    seg_len = sum(sg.length for sg in solved["result"].segments)
    assert tr.length <= seg_len + 1e-6
    assert tr.length == pytest.approx(seg_len, rel=1e-3)


def test_reverse_case_uses_reverse(planner):
    r = planner.plan(open_map(), (10.0, 7.5, 0.0), (5.0, 6.5, 0.0))
    assert r.status == PlanStatus.SUCCESS
    fwd, rev = r.trajectory.directional_lengths()
    assert rev > 3.0, "goal straight behind the vehicle should be reached mostly in reverse"


def test_start_equals_goal(planner):
    r = planner.plan(open_map(), (8.0, 7.5, 0.2), (8.0, 7.5, 0.2))
    assert r.status == PlanStatus.SUCCESS
    assert len(r.trajectory) >= 1
