"""Tests for heuristic.py."""
import math

import numpy as np
import pytest

from collision_checker import CollisionChecker
from heuristic import (HEURISTIC_MODES, CombinedHeuristic, EuclideanHeuristic, EuclidYawHeuristic,
                       ObstacleHeuristic, ReedsSheppHeuristic, build_heuristic)
from occupancy_grid import OccupancyGrid
from reeds_shepp import rs_shortest_length
from vehicle import default_vehicle

MARGIN = 0.1


def _edt(grid, vehicle):
    return CollisionChecker(grid, vehicle, "rectangle", MARGIN, 4).edt


def _parking_map():
    g = OccupancyGrid.empty(20.0, 15.0, 0.1)
    g.add_border(0.2)
    g.add_box(4.0, 0.2, 6.0, 4.0)
    g.add_box(12.0, 10.0, 16.0, 14.8)
    return g


def _all_heuristics(grid, goal, vehicle):
    R = vehicle.min_turning_radius
    edt = _edt(grid, vehicle)
    hs = {
        "euclidean": EuclideanHeuristic(goal),
        "euclid_yaw": EuclidYawHeuristic(goal, R),
        "rs_length": ReedsSheppHeuristic(goal, R),
        "obstacle": ObstacleHeuristic(grid, goal, vehicle, MARGIN, edt, 0.2),
    }
    for mode in HEURISTIC_MODES:
        hs[f"build_{mode}"] = build_heuristic(mode, grid, goal, vehicle, MARGIN, edt, 0.2)[0]
    return hs


def test_goal_gives_zero(vehicle):
    g = _parking_map()
    for goal in [(10.0, 7.5, 0.0), (8.3, 6.1, 2.0), (15.05, 4.95, -math.pi)]:
        for name, h in _all_heuristics(g, goal, vehicle).items():
            assert h.single(*goal) == pytest.approx(0.0, abs=1e-9), name
            # the same position with the opposite heading is not the goal (except pure-position heuristics)
            if "euclidean" not in name and name != "obstacle":
                assert h.single(goal[0], goal[1], goal[2] + math.pi) > 1.0, name


def test_build_heuristic_types(vehicle):
    g = _parking_map()
    edt = _edt(g, vehicle)
    goal = (10.0, 7.5, 0.0)
    for mode in HEURISTIC_MODES:
        h, obs = build_heuristic(mode, g, goal, vehicle, MARGIN, edt)
        assert (obs is not None) == (mode in ("obstacle", "rs_obstacle", "lattice_obstacle"))
        if obs is not None:
            assert isinstance(h, CombinedHeuristic) and isinstance(obs, ObstacleHeuristic)
    with pytest.raises(ValueError):
        build_heuristic("magic", g, goal, vehicle, MARGIN, edt)


def test_combined_is_elementwise_max(vehicle, rng):
    g = _parking_map()
    goal = (10.0, 7.5, 0.5)
    parts = [EuclideanHeuristic(goal), ReedsSheppHeuristic(goal, vehicle.min_turning_radius),
             ObstacleHeuristic(g, goal, vehicle, MARGIN, _edt(g, vehicle))]
    P = np.column_stack([rng.uniform(1, 19, 300), rng.uniform(1, 14, 300), rng.uniform(-3, 3, 300)])
    np.testing.assert_array_equal(CombinedHeuristic(parts)(P), np.max([p(P) for p in parts], axis=0))


def _mirror(poses, goal):
    """Reflect poses about the line through the goal along its heading."""
    gx, gy, th = goal
    c, s = math.cos(th), math.sin(th)
    dx, dy = poses[:, 0] - gx, poses[:, 1] - gy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    ly = -ly
    out = np.empty_like(poses)
    out[:, 0] = gx + c * lx - s * ly
    out[:, 1] = gy + s * lx + c * ly
    out[:, 2] = 2 * th - poses[:, 2]
    return out


def test_rs_mirror_symmetry(vehicle, rng):
    goal = (10.0, 7.5, 0.4)
    h = ReedsSheppHeuristic(goal, vehicle.min_turning_radius)
    P = np.column_stack([rng.uniform(-10, 30, 1000), rng.uniform(-10, 25, 1000), rng.uniform(-4, 4, 1000)])
    np.testing.assert_allclose(h(_mirror(P, goal)), h(P), rtol=1e-5, atol=1e-5)
    # the exact RS distance is mirror-symmetric as well
    M = _mirror(P[:30], goal)
    for p, m in zip(P[:30], M):
        R = vehicle.min_turning_radius
        assert rs_shortest_length(tuple(p), goal, R) == pytest.approx(rs_shortest_length(tuple(m), goal, R),
                                                                      rel=1e-6, abs=1e-6)


def test_obstacle_heuristic_mirror_symmetric_map(vehicle):
    # 0.125 / 0.375 m are exact binary fractions; 153 rows -> 51 coarse rows, centre row on the axis
    res, cres = 0.125, 0.375
    data = np.zeros((153, 160), dtype=np.uint8)
    data[:3, :] = data[-3:, :] = 1
    data[:, :3] = data[:, -3:] = 1
    data[20:60, 40:50] = 1          # obstacle below the axis ...
    data[60:76, 100:104] = 1
    data = data | data[::-1]        # ... mirrored above it
    g = OccupancyGrid(data, res)
    yc = 153 * res / 2.0
    goal = (15.0 + cres / 2, yc, 0.0)
    obs = ObstacleHeuristic(g, goal, vehicle, MARGIN, _edt(g, vehicle), cres)
    assert obs.ny == 51
    xs = (np.arange(obs.nx) + 0.5) * cres
    dys = (np.arange(1, 25)) * cres
    X, DY = np.meshgrid(xs, dys)
    up = np.column_stack([X.ravel(), yc + DY.ravel(), np.zeros(X.size)])
    down = np.column_stack([X.ravel(), yc - DY.ravel(), np.zeros(X.size)])
    hu, hd = obs(up), obs(down)
    assert np.isfinite(hu).sum() > 100
    np.testing.assert_allclose(hu, hd, rtol=1e-9, atol=1e-9)


def test_obstacle_free_rs_sanity(vehicle, rng):
    R = vehicle.min_turning_radius
    goal = (10.0, 7.5, 0.3)
    h = ReedsSheppHeuristic(goal, R)
    P = np.column_stack([rng.uniform(-5, 25, 1500), rng.uniform(-5, 20, 1500), rng.uniform(-math.pi, math.pi, 1500)])
    hv = h(P)
    exact = np.array([rs_shortest_length(tuple(p), goal, R) for p in P])
    eucl = np.hypot(P[:, 0] - goal[0], P[:, 1] - goal[1])
    assert np.all(exact >= eucl - 1e-9)
    # table value (minus documented slack) is optimistic and close to the exact RS length
    assert np.all(hv <= exact + 1e-6), f"RS table overestimates by {np.max(hv - exact)}"
    assert np.all(np.abs(hv - exact) <= 0.1 * R + h.slack + 0.05), f"max err {np.max(np.abs(hv - exact))}"
    # table + slack is at least euclidean distance (up to interpolation)
    assert np.all(hv + h.slack + 0.1 * R >= eucl)


def test_obstacle_free_rs_far_fallback(vehicle):
    """Outside the lookup table the heuristic falls back to max(euclid, R|dyaw|)."""
    R = vehicle.min_turning_radius
    goal = (0.0, 0.0, 0.0)
    h = ReedsSheppHeuristic(goal, R)
    far = np.array([[-60.0, 0.0, 0.0], [0.0, 80.0, 1.0]])
    v = h(far)
    assert np.all(np.isfinite(v))
    exact = [rs_shortest_length(tuple(p), goal, R) for p in far]
    assert np.all(v <= np.array(exact) + 1e-6)
    assert v[0] == pytest.approx(60.0 - h.slack, abs=1e-6)


def test_obstacle_heuristic_lower_bound_with_obstacles(vehicle):
    g = _parking_map()
    res = 0.2
    goal = (10.1, 7.5, 0.0)  # coarse cell centre
    obs = ObstacleHeuristic(g, goal, vehicle, MARGIN, _edt(g, vehicle), res)
    xs = (np.arange(obs.nx) + 0.5) * res
    ys = (np.arange(obs.ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    h = obs(P)
    fin = np.isfinite(h)
    assert fin.sum() > 1000
    eucl = np.hypot(P[:, 0] - goal[0], P[:, 1] - goal[1])
    assert np.all(h[fin] >= eucl[fin] - obs.offset - 1e-9)
    # arbitrary (non cell-centre) positions: extra res/sqrt(2) quantisation slack
    rng = np.random.default_rng(5)
    Q = np.column_stack([rng.uniform(0, 20, 2000), rng.uniform(0, 15, 2000), np.zeros(2000)])
    hq = obs(Q)
    f = np.isfinite(hq)
    eq = np.hypot(Q[:, 0] - goal[0], Q[:, 1] - goal[1])
    assert np.all(hq[f] >= eq[f] - obs.offset - res * math.sqrt(2) / 2 - 1e-9)


def test_obstacle_heuristic_wall_increases_value(vehicle):
    g = OccupancyGrid.empty(20.0, 15.0, 0.1)
    g.add_border(0.2)
    free = ObstacleHeuristic(g, (15.0, 7.5, 0.0), vehicle, MARGIN, _edt(g, vehicle))
    g.add_box(9.5, 0.0, 10.5, 12.5)   # wall with a gap only near the top
    obs = ObstacleHeuristic(g, (15.0, 7.5, 0.0), vehicle, MARGIN, _edt(g, vehicle))
    start = (5.0, 7.5)
    eucl = 10.0
    assert free.single(*start, 0.0) == pytest.approx(eucl, abs=3 * free.offset)
    hv = obs.single(*start, 0.0)
    assert math.isfinite(hv)
    assert hv > eucl + 5.0, f"wall should make the heuristic much larger than euclidean ({hv})"
    assert obs.reachable(*start)


def test_unreachable_goal(vehicle):
    g = OccupancyGrid.empty(20.0, 15.0, 0.1)
    g.add_border(0.2)
    # closed ring around the goal
    g.add_box(11.0, 4.0, 19.0, 4.3)
    g.add_box(11.0, 10.7, 19.0, 11.0)
    g.add_box(11.0, 4.0, 11.3, 11.0)
    g.add_box(18.7, 4.0, 19.0, 11.0)
    goal = (14.0, 7.5, 0.0)
    edt = _edt(g, vehicle)
    obs = ObstacleHeuristic(g, goal, vehicle, MARGIN, edt)
    assert obs.single(*goal) == pytest.approx(0.0)
    assert math.isinf(obs.single(4.0, 7.5, 0.0))
    assert not obs.reachable(4.0, 7.5)
    assert obs.reachable(15.0, 8.0)
    h, o = build_heuristic("rs_obstacle", g, goal, vehicle, MARGIN, edt)
    assert math.isinf(h.single(4.0, 7.5, 0.0))
    # goal outside the map: no crash, nothing reachable
    out = ObstacleHeuristic(g, (50.0, 50.0, 0.0), vehicle, MARGIN, edt)
    assert not out.goal_valid and not out.reachable(4.0, 7.5)
    # query outside the map -> inf
    assert math.isinf(obs.single(-5.0, -5.0, 0.0))


def test_values_finite_and_non_negative(vehicle, rng):
    g = _parking_map()
    goal = (10.0, 7.5, 1.0)
    P = np.column_stack([rng.uniform(0.5, 19.5, 2000), rng.uniform(0.5, 14.5, 2000),
                         rng.uniform(-2 * math.pi, 2 * math.pi, 2000)])
    for name, h in _all_heuristics(g, goal, vehicle).items():
        v = h(P)
        assert v.shape == (len(P),)
        assert np.all(v >= 0.0), name
        assert not np.any(np.isnan(v)), name
        if "obstacle" not in name:
            assert np.all(np.isfinite(v)), name
    # obstacle heuristic is finite in the connected free space around the goal
    obs = ObstacleHeuristic(g, goal, vehicle, MARGIN, _edt(g, vehicle))
    open_pts = np.array([[8.0, 7.0, 0.0], [15.0, 5.0, 0.0], [3.0, 10.0, 0.0]])
    assert np.all(np.isfinite(obs(open_pts)))
