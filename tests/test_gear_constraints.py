"""Manoeuvre structure: gear in the state, hard gear-change limit, minimum
manoeuvre length, gear-change-aware heuristic table."""
import math

import numpy as np
import pytest

from lattice_heuristic import LatticeHLUT
from motion_primitives import PrimitiveConfig, generate_primitives
from occupancy_grid import OccupancyGrid
from path_shortcut import count_switches, min_run_length
from state_lattice import PlannerConfig, PlanStatus, StateLatticePlanner
from vehicle import default_vehicle


def _runs(traj):
    return [p.points[-1].arc_length - p.points[0].arc_length for p in traj.split_by_direction()]


def _reverse_parking_scene():
    g = OccupancyGrid.empty(30, 22, 0.1)
    g.add_box(0, 0, 30, 1.0)
    for x in (12.4, 17.6):
        g.add_rectangle(x, 4.0, 4.7, 1.85, math.pi / 2)
    return g, (8.0, 12.0, 0.0), (15.0, 3.2, math.pi / 2)


def test_gear_is_part_of_the_state():
    pl = StateLatticePlanner(default_vehicle())
    g = OccupancyGrid.empty(10, 10, 0.1)
    assert pl.state_key(5, 5, 0, 1, g) != pl.state_key(5, 5, 0, -1, g)


def test_gear_change_limit_is_hard():
    g, s, goal = _reverse_parking_scene()
    v = default_vehicle()
    none = StateLatticePlanner(v, PlannerConfig(max_gear_switches=0, max_planning_time=20,
                                                max_expanded_nodes=20000)).plan(g, s, goal)
    assert not none.success            # reversing into the slot needs a gear change
    one = StateLatticePlanner(v, PlannerConfig(max_gear_switches=1, max_planning_time=20)).plan(g, s, goal)
    assert one.success and one.trajectory.n_direction_changes <= 1


@pytest.mark.parametrize("mn", [0.5, 1.0, 1.5])
def test_minimum_manoeuvre_length_is_hard(mn):
    g, s, goal = _reverse_parking_scene()
    v = default_vehicle()
    cfg = PlannerConfig(min_maneuver_length=mn, min_slot_maneuver_length=mn,
                        preferred_maneuver_length=max(mn, 2.0), max_planning_time=30)
    res = StateLatticePlanner(v, cfg).plan(g, s, goal)
    assert res.success, res.message
    if res.trajectory.n_direction_changes > 0:
        assert min(_runs(res.trajectory)) >= mn - 1e-6


def test_maneuver_rules():
    from maneuver import ManeuverRules
    cfg = PlannerConfig(max_gear_switches=2, min_maneuver_length=1.0, min_slot_maneuver_length=0.4,
                        slot_zone_radius=2.0, preferred_maneuver_length=2.0)
    r = ManeuverRules(cfg, default_vehicle(), (0.0, 0.0, 0.0), cfg.max_gear_switches)
    far, near = (10.0, 0.0), (1.0, 0.0)
    piece = lambda d, l, x, y, s=0: (d, l, [s], x, y)
    # a path's first manoeuvre is measured from its start (regression: it was 'inf')
    assert not r.evaluate([piece(1, 0.5, *far), piece(-1, 3.0, *near)], 9.5, 0.0, 0, math.inf, 0, 0, True,
                          final=True)[0]
    # in the aisle a manoeuvre must be >= 1.0 m before changing gear, near the slot >= 0.4 m
    assert r.evaluate([piece(1, 1.2, *far), piece(-1, 3.0, *near)], 8.8, 0.0, 0, math.inf, 0, 0, True,
                      final=True)[0]
    assert r.evaluate([piece(-1, 0.5, *near), piece(1, 0.5, 0.5, 0.0)], 1.5, 0.0, -1, 3.0, 1, 0, True,
                      final=True)[0]
    # gear-change limit and final manoeuvre length
    many = [piece(1, 2.0, *far), piece(-1, 2.0, *far), piece(1, 2.0, *far), piece(-1, 2.0, *near)]
    assert not r.evaluate(many, 12.0, 0.0, 0, math.inf, 0, 0, True, final=True)[0]
    assert not r.evaluate([piece(1, 2.0, *near), piece(-1, 0.2, 0.8, 0.0)], 3.0, 0.0, 0, math.inf, 0, 0, True,
                          final=True)[0]
    # steering reversals within one manoeuvre are penalised
    ok, soft = r.evaluate([piece(1, 2.0, 2.0, 0.0, 1), piece(1, 2.0, 4.0, 0.0, -1)], 0.0, 0.0, 0, math.inf,
                          0, 0, True, final=True)
    assert ok and soft == pytest.approx(cfg.steer_reversal_penalty)


@pytest.mark.skipif(not LatticeHLUT.available(), reason="numba not available")
def test_hlut_levels_are_monotone():
    pset = generate_primitives(default_vehicle(), PrimitiveConfig())
    t = LatticeHLUT.get(pset, 1.5, 0.3, 10.0, extent=4.0, resolution=0.25, switch_levels=3).table
    # more remaining gear changes can only make the cost-to-go smaller
    assert np.all(t[1] <= t[0] + 1e-9) and np.all(t[2] <= t[1] + 1e-9)
    # a goal straight behind needs no change in reverse, but forward-arrival
    # states that must reverse need one: level 0 is inf where level 1 is finite
    assert np.isinf(t[0]).sum() > np.isinf(t[1]).sum()


def test_sealed_goal_is_rejected_fast():
    import time
    from scenarios import build_scenario
    from parking_planner import BidirectionalPlanner
    sc = build_scenario("C4")
    v = sc.make_vehicle()
    t = time.perf_counter()
    res = BidirectionalPlanner(v, sc.make_config(PlannerConfig(max_planning_time=60))).plan(sc.grid, sc.start, sc.goal)
    assert res.status == PlanStatus.NO_PATH and "sealed" in res.message
    assert time.perf_counter() - t < 10.0
