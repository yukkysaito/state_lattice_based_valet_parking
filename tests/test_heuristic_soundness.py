"""Soundness of the obstacle heuristic's unreachability proof.

The planner returns UNREACHABLE (and prunes successors) when the 2D obstacle
heuristic is infinite. A false "infinite" would silently reject a feasible
parking manoeuvre, so these tests check the property the proof relies on:
every collision-free pose lies in a free heuristic cell, and every pose of a
feasible path has a finite heuristic value.
"""
import math

import numpy as np
import pytest

from collision_checker import CollisionChecker
from heuristic import ObstacleHeuristic
from occupancy_grid import OccupancyGrid
from state_lattice import PlannerConfig, PlanStatus, StateLatticePlanner
from vehicle import default_vehicle, large_suv

MARGIN = 0.1


def _random_map(rng, w=24.0, h=18.0):
    g = OccupancyGrid.empty(w, h, 0.1)
    for _ in range(int(rng.integers(6, 14))):
        g.add_rectangle(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.3, 4.0), rng.uniform(0.2, 2.0),
                        rng.uniform(-math.pi, math.pi))
    return g


@pytest.mark.parametrize("seed", range(6))
def test_collision_free_poses_are_in_free_cells(seed):
    rng = np.random.default_rng(seed)
    v = default_vehicle()
    g = _random_map(rng)
    cc = CollisionChecker(g, v, "rectangle", MARGIN)
    goal = (12.0, 9.0, 0.0)
    obs = ObstacleHeuristic(g, goal, v, MARGIN, cc.edt, 0.2)
    poses = np.column_stack([rng.uniform(0, 24, 4000), rng.uniform(0, 18, 4000), rng.uniform(-3.14, 3.14, 4000)])
    free_poses = poses[~cc.check_poses(poses)]
    assert len(free_poses) > 50
    ix = np.floor((free_poses[:, 0] - obs.origin[0]) / obs.res).astype(int)
    iy = np.floor((free_poses[:, 1] - obs.origin[1]) / obs.res).astype(int)
    assert obs.free[iy, ix].all(), "a collision-free pose fell into a blocked heuristic cell"


@pytest.mark.parametrize("angle_deg", [0, 17, 30, 45, 62, 90])
@pytest.mark.parametrize("vehicle_fn", [default_vehicle, large_suv])
def test_narrow_corridor_not_rejected(angle_deg, vehicle_fn):
    """Corridor barely wider than the inflated vehicle, at arbitrary angles."""
    v = vehicle_fn()
    width = v.vehicle_width + 2 * MARGIN + 0.02  # carving marks partially free cells free
    a = math.radians(angle_deg)
    g = OccupancyGrid.empty(40, 40, 0.1)
    g.data[:] = 1
    c, s = math.cos(a), math.sin(a)
    center = np.array([20.0, 20.0])
    # carve a straight corridor (free) of the given width, 30 m long
    corridor = np.array([[-15, -width / 2], [15, -width / 2], [15, width / 2], [-15, width / 2]])
    rot = np.array([[c, -s], [s, c]])
    g.add_polygon(corridor @ rot.T + center, value=0)
    # rasterisation of the carve keeps a cell only if fully free: re-mark partial cells
    cc = CollisionChecker(g, v, "rectangle", MARGIN)
    start = (center[0] - 8 * c, center[1] - 8 * s, a)
    goal = (center[0] + 8 * c, center[1] + 8 * s, a)
    if cc.check_pose(*start) or cc.check_pose(*goal):
        pytest.skip("rasterisation made the corridor too narrow for this angle")
    # the straight motion itself must be collision free for the test to be meaningful
    t = np.linspace(0, 1, 400)[:, None]
    line = np.hstack([np.array(start[:2]) + t * (np.array(goal[:2]) - np.array(start[:2])), np.full((400, 1), a)])
    if cc.check_poses(line).any():
        pytest.skip("straight motion collides after rasterisation")
    obs = ObstacleHeuristic(g, goal, v, MARGIN, cc.edt, 0.2)
    h = obs(line)
    assert np.all(np.isfinite(h)), f"feasible corridor rejected at {angle_deg} deg"


@pytest.mark.parametrize("sid", ["B3", "E3", "E8", "F1", "C2"])
def test_feasible_paths_have_finite_obstacle_heuristic(sid):
    """Plan WITHOUT the obstacle heuristic (no pruning) on narrow scenes; every
    pose of the found path must have a finite obstacle-heuristic value, and the
    default planner must not claim unreachability."""
    from scenarios import build_scenario
    sc = build_scenario(sid)
    v = sc.make_vehicle()
    cfg = sc.make_config(PlannerConfig(heuristic="reeds_shepp", max_planning_time=60.0,
                                       endpoint_relaxation=False))
    from parking_planner import BidirectionalPlanner
    planner = StateLatticePlanner(v, cfg)
    res = BidirectionalPlanner(v, cfg, shortcut=False).plan(sc.grid, sc.start, sc.goal)
    assert res.success, res.message
    cc = planner.make_checker(sc.grid)
    obs = ObstacleHeuristic(sc.grid, sc.goal, v, planner.check_margin, cc.edt, cfg.obstacle_heuristic_resolution)
    assert np.all(np.isfinite(obs(res.trajectory.poses())))
    res2 = StateLatticePlanner(v, sc.make_config(PlannerConfig(max_planning_time=5.0))).plan(sc.grid, sc.start, sc.goal)
    assert res2.status != PlanStatus.UNREACHABLE
