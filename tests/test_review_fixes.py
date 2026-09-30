"""Regression tests for the findings of the independent safety review."""
import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from collision_checker import CollisionChecker
from lattice_heuristic import LatticeHLUT
from motion_primitives import PrimitiveConfig, generate_primitives
from occupancy_grid import OccupancyGrid
from state_lattice import PlannerConfig, StateLatticePlanner
from vehicle import default_vehicle, small_car


@pytest.mark.skipif(not LatticeHLUT.available(), reason="numba not available")
def test_hlut_equals_float64_reference_dijkstra():
    """The table must be the exact lattice cost-to-go (float32/float64 mix-up bug)."""
    import heapq
    pset = generate_primitives(default_vehicle(), PrimitiveConfig())
    ext, res = 3.0, 0.25
    t = LatticeHLUT.get(pset, 1.5, 0.3, 10.0, extent=ext, resolution=res, goal_radius=0.15).table[0]  # unlimited level
    # pure-python reference on the same snapped graph
    n = t.shape[1]
    n_yaw = t.shape[3]
    half = (n - 1) // 2
    cfg = pset.config
    cost = pset.lengths * np.where(pset.directions > 0, 1.0, 1.5) + 0.3 * pset.lengths * np.abs(pset.steerings) / \
        pset.vehicle.max_steer_angle
    dyaw = np.round(pset.ends[:, 2] / cfg.yaw_resolution).astype(int)
    ref = np.full(t.shape, np.inf)
    heap = []
    rg = int(math.ceil(0.15 / res))
    for ix in range(half - rg, half + rg + 1):
        for iy in range(half - rg, half + rg + 1):
            if ((ix - half) * res) ** 2 + ((iy - half) * res) ** 2 <= 0.15 ** 2 + 1e-9:
                for d in range(2):
                    ref[d, ix, iy, 0] = 0.0
                    heapq.heappush(heap, (0.0, d, ix, iy, 0))
    while heap:
        h, d, ix, iy, iw = heapq.heappop(heap)
        if h > ref[d, ix, iy, iw]:
            continue
        for p in range(len(pset)):
            dp = 0 if pset.directions[p] > 0 else 1
            if dp != d:
                continue
            iw0 = (iw - dyaw[p]) % n_yaw
            yaw0 = iw0 * cfg.yaw_resolution
            c, s = math.cos(yaw0), math.sin(yaw0)
            jx = ix - int(round((c * pset.ends[p, 0] - s * pset.ends[p, 1]) / res))
            jy = iy - int(round((s * pset.ends[p, 0] + c * pset.ends[p, 1]) / res))
            if not (0 <= jx < n and 0 <= jy < n):
                continue
            for dprev in range(2):
                v = h + cost[p] + (10.0 if dprev != dp else 0.0)
                if v < ref[dprev, jx, jy, iw0]:
                    ref[dprev, jx, jy, iw0] = v
                    heapq.heappush(heap, (v, dprev, jx, jy, iw0))
    finite = np.isfinite(ref)
    assert np.array_equal(finite, np.isfinite(t))
    assert np.max(np.abs(ref[finite] - t[finite]) / np.maximum(ref[finite], 1.0)) < 1e-6  # float32 storage


def test_yaw_resolution_must_divide_two_pi():
    with pytest.raises(ValueError):
        PlannerConfig(yaw_resolution=math.radians(7.0)).validate()
    PlannerConfig(yaw_resolution=math.radians(2.5)).validate()


@pytest.mark.parametrize("use_numba", [True, False])
def test_nan_poses_are_collisions(use_numba):
    g = OccupancyGrid(np.ones((60, 60)), 0.1)
    cc = CollisionChecker(g, default_vehicle(), use_numba=use_numba)
    assert cc.check_poses(np.array([[np.nan, 3, 0], [3, np.nan, 0], [3, 3, np.nan]])).all()
    g0 = OccupancyGrid.empty(20, 20, 0.1)
    cc0 = CollisionChecker(g0, default_vehicle(), use_numba=use_numba)
    assert cc0.check_poses(np.array([[np.nan, 10, 0]])).all()


@pytest.mark.parametrize("use_numba", [True, False])
def test_footprint_inside_thick_obstacle_with_coarse_cells(use_numba):
    """numpy fallback used to miss this (exact stage only saw boundary cells)."""
    v = small_car()
    g = OccupancyGrid.empty(40, 40, 1.0)
    g.add_box(5, 5, 35, 35)
    cc = CollisionChecker(g, v, margin=0.0, use_numba=use_numba)
    assert cc.check_pose(20.0, 20.0, 0.3)
    assert cc.min_clearance(np.array([[20.0, 20.0, 0.3]]))[0] == 0.0


@pytest.mark.parametrize("use_numba", [True, False])
def test_min_clearance_is_exact(use_numba):
    rng = np.random.default_rng(3)
    v = default_vehicle()
    g = OccupancyGrid.empty(20, 20, 0.1)
    for _ in range(12):
        g.add_rectangle(rng.uniform(0, 20), rng.uniform(0, 20), rng.uniform(0.2, 2), rng.uniform(0.2, 2),
                        rng.uniform(-3, 3))
    cc = CollisionChecker(g, v, margin=0.0, use_numba=use_numba)
    iy, ix = np.nonzero(g.data)
    boxes = [box(x * 0.1, y * 0.1, (x + 1) * 0.1, (y + 1) * 0.1) for x, y in zip(ix, iy)]
    poses = np.column_stack([rng.uniform(3, 17, 60), rng.uniform(3, 17, 60), rng.uniform(-3.1, 3.1, 60)])
    got = cc.min_clearance(poses, cap=2.0)
    for p, m in zip(poses, got):
        fp = Polygon(v.footprint_world(*p))
        truth = min(2.0, min(fp.distance(b) for b in boxes))
        assert m == pytest.approx(truth, abs=1e-9)


def test_roi_crop_preserves_world_coordinates():
    g = OccupancyGrid.empty(100, 80, 0.1)
    g.add_box(40, 30, 41, 31)
    sub = g.crop(30, 20, 60, 50)
    assert sub.width < g.width and sub.height < g.height
    assert sub.is_occupied(40.5, 30.5) and not sub.is_occupied(45, 45)
    assert sub.is_occupied(10, 10)  # outside the ROI counts as occupied


def test_large_map_setup_is_bounded():
    import time
    g = OccupancyGrid.empty(300, 300, 0.1)
    pl = StateLatticePlanner(default_vehicle(), PlannerConfig(max_planning_time=0.05))
    t = time.perf_counter()
    pl.plan(g, (140, 150, 0), (160, 150, 0))
    assert time.perf_counter() - t < 2.0
