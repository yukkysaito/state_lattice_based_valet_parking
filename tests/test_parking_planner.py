"""Staged facade, exit planning (path reversal) and micro primitives."""
import math

import numpy as np

from metrics import IndependentValidator
from motion_primitives import PrimitiveConfig, generate_primitives
from occupancy_grid import OccupancyGrid
from parking_planner import BidirectionalPlanner, plan_by_exit, reverse_trajectory
from state_lattice import PlannerConfig, PlanStatus
from trajectory import Trajectory
from vehicle import default_vehicle


def test_reverse_trajectory_flips_direction_and_order():
    v = default_vehicle()
    poses = np.array([[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [0.6, 0, 0]], float)
    t = Trajectory.from_arrays(poses, [1, 1, 1, -1], [0.0, 0.1, 0.1, -0.2], v.wheel_base)
    r = reverse_trajectory(t, v.wheel_base)
    assert np.allclose(r.poses(), poses[::-1])
    # motion 1.0 -> 0.6 was reverse; traversed backwards (0.6 -> 1.0) it is forward
    assert r.directions().tolist() == [1, 1, -1, -1]
    assert r.n_direction_changes == t.n_direction_changes


def test_exit_planning_starts_at_start_and_is_collision_free():
    v = default_vehicle()
    g = OccupancyGrid.empty(30, 20, 0.1)
    g.add_box(12, 0, 13, 6)
    start, goal = (5.0, 10.0, 0.0), (20.0, 4.0, math.pi / 2)
    res = plan_by_exit(v, PlannerConfig(max_planning_time=10), g, start, goal)
    assert res.success, res.message
    p0, pn = res.trajectory.points[0], res.trajectory.points[-1]
    assert math.hypot(p0.x - start[0], p0.y - start[1]) < 1e-2
    assert math.hypot(pn.x - goal[0], pn.y - goal[1]) < 1e-6
    assert IndependentValidator(g, v).is_collision_free(res.trajectory.poses())


def test_bidirectional_reports_proofs_for_the_original_problem():
    v = default_vehicle()
    g = OccupancyGrid.empty(20, 15, 0.1)
    g.add_box(9, 4, 12, 7)
    bp = BidirectionalPlanner(v, PlannerConfig(max_planning_time=5))
    assert bp.plan(g, (10.5, 5.5, 0), (3, 3, 0)).status == PlanStatus.START_IN_COLLISION
    assert bp.plan(g, (3, 3, 0), (10.5, 5.5, 0)).status == PlanStatus.GOAL_IN_COLLISION


def test_tight_parallel_rejected_fast_under_product_constraints():
    """1.3 x L parallel slot: impossible with <= 3 gear changes and >= 0.4 m
    manoeuvres at a 0.1 m margin -> fast 'goal region sealed' answer."""
    from scenarios import build_scenario
    import time
    sc = build_scenario("C3")
    v = sc.make_vehicle()
    t = time.perf_counter()
    res = BidirectionalPlanner(v, sc.make_config(PlannerConfig(max_planning_time=60))).plan(sc.grid, sc.start, sc.goal)
    assert res.status == PlanStatus.NO_PATH and "sealed" in res.message
    assert time.perf_counter() - t < 15.0


def test_tight_parallel_solved_without_manoeuvre_limits():
    """Same slot without the limits: the exit search solves it; the reversed path
    starts at the start pose."""
    from scenarios import build_scenario
    sc = build_scenario("C3")
    v = sc.make_vehicle()
    cfg = sc.make_config(PlannerConfig(max_planning_time=60, max_gear_switches=None, min_maneuver_length=0.2,
                                       min_slot_maneuver_length=0.2,
                                       preferred_maneuver_length=0.2))
    res = BidirectionalPlanner(v, cfg).plan(sc.grid, sc.start, sc.goal)
    assert res.success, res.message
    assert res.search == "exit"
    p0, pn = res.trajectory.points[0], res.trajectory.points[-1]
    assert math.hypot(p0.x - sc.start[0], p0.y - sc.start[1]) < 1e-2
    assert math.hypot(pn.x - sc.goal[0], pn.y - sc.goal[1]) <= 0.15
    assert IndependentValidator(sc.grid, v).is_collision_free(res.trajectory.poses())


def test_bidirectional_is_deterministic():
    from scenarios import build_scenario
    sc = build_scenario("C2")
    v = sc.make_vehicle()
    bp = BidirectionalPlanner(v, sc.make_config(PlannerConfig(max_planning_time=20)))
    a, b = bp.plan(sc.grid, sc.start, sc.goal), bp.plan(sc.grid, sc.start, sc.goal)
    assert a.success and b.success and a.search == b.search
    assert np.array_equal(a.trajectory.poses(), b.trajectory.poses())


def test_micro_primitives_are_short_straight_or_full_lock():
    v = default_vehicle()
    ps = generate_primitives(v, PrimitiveConfig(length_classes=("micro", "short")))
    micro = [p for p in ps if p.length_class == "micro"]
    assert len(micro) == 6  # {straight, +max, -max} x {fwd, rev}
    for p in micro:
        assert p.length <= 0.3 + 1e-9
        assert abs(p.steering) < 1e-9 or abs(abs(p.steering) - v.max_steer_angle) < 1e-9
