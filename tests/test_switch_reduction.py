"""Direction-switch reduction: exit search over the goal tolerance region, RS
shortcutting, minimum run length, separate longitudinal / lateral margins."""
import math

import numpy as np
import pytest

from collision_checker import CollisionChecker
from metrics import IndependentValidator
from occupancy_grid import OccupancyGrid
from parking_planner import BidirectionalPlanner, goal_region_seeds
from path_shortcut import count_switches, min_run_length, path_cost, shortcut
from state_lattice import PlannerConfig, StateLatticePlanner
from vehicle import default_vehicle

TIGHT = ["B4", "D2", "N-021", "N-030", "N-094"]


def _plan(sid):
    from scenarios import build_scenario
    sc = build_scenario(sid)
    v = sc.make_vehicle()
    cfg = sc.make_config(PlannerConfig(max_planning_time=60))
    return sc, v, cfg, BidirectionalPlanner(v, cfg).plan(sc.grid, sc.start, sc.goal)


@pytest.mark.parametrize("sid", TIGHT)
def test_bidirectional_paths_are_clean(sid):
    sc, v, cfg, res = _plan(sid)
    assert res.success, res.message
    t = res.trajectory
    # manoeuvre rules (length per place, gear-change limit) hold for the final path
    from maneuver import ManeuverRules, path_ok
    rules = ManeuverRules(cfg, v, sc.goal, cfg.max_gear_switches)
    assert path_ok(rules, t.poses(), t.directions(), t.steerings())[0]
    assert min_run_length(t.poses(), t.directions()) >= cfg.min_slot_maneuver_length - 1e-6
    # post-processing never adds a switch
    raw = res.raw_trajectory or t
    assert t.n_direction_changes <= raw.n_direction_changes
    # ends inside the goal tolerance, starts exactly at the start
    p0, pn = t.points[0], t.points[-1]
    assert math.hypot(p0.x - sc.start[0], p0.y - sc.start[1]) < 1e-2
    assert math.hypot(pn.x - sc.goal[0], pn.y - sc.goal[1]) <= cfg.position_tolerance + 1e-6
    yaw_err = abs(math.atan2(math.sin(pn.yaw - sc.goal[2]), math.cos(pn.yaw - sc.goal[2])))
    assert yaw_err <= cfg.yaw_tolerance + 1e-6
    # still collision-free with the full margin (independent check, 1 cm)
    val = IndependentValidator(sc.grid, v, margin=cfg.safety_margin - 1e-3)
    assert not val.colliding_indices(t.poses(), densify_step=0.01)


def test_shortcut_never_increases_cost_or_switches():
    sc, v, cfg, res = _plan("N-015")
    pl = StateLatticePlanner(v, cfg)
    chk, _ = pl.make_zoned_checker(pl.region_of_interest(sc.grid, sc.start, sc.goal),
                                   (("start", sc.start), ("goal", sc.goal)))
    raw = res.raw_trajectory or res.trajectory
    new = shortcut(pl, chk, raw)
    assert new.n_direction_changes <= raw.n_direction_changes
    c_raw = path_cost(pl, chk, raw.poses(), raw.directions(), raw.steerings())
    c_new = path_cost(pl, chk, new.poses(), new.directions(), new.steerings())
    assert c_new <= c_raw + 1e-9
    assert not chk.check_poses(new.poses()).any()


def test_goal_region_seeds_stay_in_tolerance():
    cfg = PlannerConfig()
    goal = (10.0, 5.0, 0.7)
    seeds = goal_region_seeds(goal, cfg)
    assert len(seeds) > 20
    for (x, y, yaw), g0 in seeds:
        assert math.hypot(x - goal[0], y - goal[1]) <= cfg.position_tolerance + 1e-9
        assert abs(yaw - goal[2]) <= cfg.yaw_tolerance + 1e-9
        assert 0 < g0 < cfg.direction_switch_penalty  # an offset pose must not look like a free switch


def test_count_switches_and_runs():
    d = np.array([1, 1, 1, -1, -1, 1])
    p = np.column_stack([np.arange(6) * 0.1, np.zeros(6), np.zeros(6)])
    assert count_switches(d) == 2
    assert min_run_length(p, d) == pytest.approx(0.1)


def test_separate_longitudinal_margin():
    """An obstacle 0.08 m in front of the bumper is rejected with a 0.1 m margin
    but accepted with a 0.05 m longitudinal margin; lateral stays strict."""
    v = default_vehicle()
    g = OccupancyGrid.empty(20, 10, 0.02)
    front = 5.0 + v.wheel_base + v.front_overhang
    g.add_box(front + 0.08, 4.0, front + 0.3, 6.0)          # 8 cm ahead of the bumper
    g.add_box(4.5, 5.0 + v.half_width + 0.08, 8.0, 7.0)      # 8 cm beside the left side
    iso = CollisionChecker(g, v, margin=0.1)
    aniso_lon = CollisionChecker(g, v, margin=(0.05, 0.1))
    pose = (5.0, 5.0, 0.0)
    assert iso.check_pose(*pose)
    assert aniso_lon.check_pose(*pose)          # the side obstacle still violates 0.1 lateral
    g2 = OccupancyGrid.empty(20, 10, 0.02)
    g2.add_box(front + 0.08, 4.0, front + 0.3, 6.0)
    assert CollisionChecker(g2, v, margin=0.1).check_pose(*pose)
    assert not CollisionChecker(g2, v, margin=(0.05, 0.1)).check_pose(*pose)
    pl = StateLatticePlanner(v, PlannerConfig(safety_margin_longitudinal=0.05))
    assert pl.check_margins[0] < pl.check_margins[1]
