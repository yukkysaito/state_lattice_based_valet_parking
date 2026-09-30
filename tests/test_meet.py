"""Meet-in-the-middle joins (forward branch + RS + reversed exit branch) are
complete, rule-conforming, collision-free paths."""
import math

import numpy as np
import pytest

from maneuver import ManeuverRules, path_ok
from metrics import IndependentValidator
from parking_planner import BidirectionalPlanner
from scenarios import build_scenario
from state_lattice import PlannerConfig


@pytest.mark.parametrize("sid", ["D3", "E8"])
def test_meet_in_the_middle_path_is_valid(sid):
    sc = build_scenario(sid)
    v = sc.make_vehicle()
    cfg = sc.make_config(PlannerConfig(max_planning_time=120))
    res = BidirectionalPlanner(v, cfg, shortcut=False).plan(sc.grid, sc.start, sc.goal)
    assert res.success and "meet-in-the-middle" in res.message, res.message
    t = res.trajectory
    poses = t.poses()
    assert np.allclose(poses[0], sc.start, atol=1e-6)
    assert math.hypot(poses[-1][0] - sc.goal[0], poses[-1][1] - sc.goal[1]) <= cfg.position_tolerance + 1e-6
    assert abs(math.remainder(poses[-1][2] - sc.goal[2], 2 * math.pi)) <= cfg.yaw_tolerance + 1e-6
    steps = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    assert steps.max() < 0.2                              # continuous (no jump at the junction)
    dirs = np.array([p.direction for p in t.points])
    steer = np.array([p.steering for p in t.points])
    rules = ManeuverRules(cfg, v, sc.goal, cfg.max_gear_switches)
    assert path_ok(rules, poses, dirs, steer)[0]
    assert IndependentValidator(sc.grid, v).is_collision_free(poses)
