"""Free-space lattice heuristic look-up table (lattice_heuristic.py)."""
import math

import numpy as np
import pytest

from lattice_heuristic import LatticeHLUT
from motion_primitives import PrimitiveConfig, generate_primitives
from occupancy_grid import OccupancyGrid
from state_lattice import PlannerConfig, StateLatticePlanner
from vehicle import default_vehicle

pytestmark = pytest.mark.skipif(not LatticeHLUT.available(), reason="numba not available")

RW, SW, SP = 1.5, 0.3, 10.0


@pytest.fixture(scope="module")
def hlut():
    pset = generate_primitives(default_vehicle(), PrimitiveConfig())
    return LatticeHLUT.get(pset, RW, SW, SP, extent=6.0, resolution=0.25, goal_radius=0.15)


def _h(t, x, y, yaw, d=0):
    return float(t.lookup(np.array([x], float), np.array([y], float), np.array([yaw], float), np.array([d]))[0])


def test_goal_is_zero(hlut):
    assert _h(hlut, 0, 0, 0) == 0.0
    assert _h(hlut, 0, 0, 0, 1) == 0.0 and _h(hlut, 0, 0, 0, -1) == 0.0


def test_straight_forward_and_reverse_costs(hlut):
    # goal 3 m ahead (pose behind the goal) -> forward 3 m; goal behind -> reverse 3 m * RW
    fwd = _h(hlut, -3.0, 0, 0)
    rev = _h(hlut, 3.0, 0, 0)
    assert fwd == pytest.approx(3.0, abs=0.35)
    assert rev == pytest.approx(3.0 * RW, abs=0.5)
    assert rev > fwd


def test_switch_penalty_depends_on_arrival_direction(hlut):
    # arriving forward at a pose that must reverse to the goal costs an extra switch
    after_fwd = _h(hlut, 3.0, 0, 0, +1)
    after_rev = _h(hlut, 3.0, 0, 0, -1)
    assert after_fwd == pytest.approx(after_rev + SP, abs=1e-3)


def test_left_right_symmetry(hlut):
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y, yaw = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.integers(-36, 36) * math.radians(5)
        a, b = _h(hlut, x, y, yaw), _h(hlut, x, -y, -yaw)
        if math.isfinite(a) and math.isfinite(b):
            assert a == pytest.approx(b, abs=0.3)


def test_outside_table_is_nan(hlut):
    assert math.isnan(_h(hlut, 50.0, 0, 0))


def test_heuristic_close_to_planned_cost_in_free_space():
    """The table approximates the lattice cost-to-go: on an empty map it should be
    close to (and not wildly above) the cost the planner actually achieves."""
    v = default_vehicle()
    cfg = PlannerConfig(max_planning_time=10.0)
    p = StateLatticePlanner(v, cfg)
    g = OccupancyGrid.empty(30, 24, 0.1)
    start, goal = (8.0, 12.0, 0.0), (16.0, 8.0, math.pi / 2)
    res = p.plan(g, start, goal)
    assert res.success
    hl = p.lattice_table()
    c, s = math.cos(goal[2]), math.sin(goal[2])
    dx, dy = start[0] - goal[0], start[1] - goal[1]
    h = float(hl.lookup(np.array([c * dx + s * dy]), np.array([-s * dx + c * dy]),
                        np.array([start[2] - goal[2]]), np.array([0]))[0])
    assert 0.6 * res.cost <= h <= 1.25 * res.cost
