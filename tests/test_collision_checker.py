"""Tests for collision_checker.py (public API only).

Geometry reference (default vehicle, margin 0, pose (5.03, 5.03, 0)):
front x = 8.72, rear x = 4.03, left y = 5.98, right y = 4.08.
Cell (ix, iy) of a 0.1 m grid with origin (0, 0) spans [ix/10, ix/10+0.1] x [iy/10, iy/10+0.1].
"""
import math

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from collision_checker import CollisionChecker
from motion_primitives import primitives_for
from occupancy_grid import OccupancyGrid
from vehicle import default_vehicle

POSE = (5.03, 5.03, 0.0)
NUMBA_MODES = [None, False, True]


def make_checker(grid, vehicle, method="rectangle", margin=0.0, n_circles=4, use_numba=None):
    """Construct a checker; ``use_numba`` is optional (internal switch, may disappear)."""
    if use_numba is None:
        return CollisionChecker(grid, vehicle, method, margin, n_circles)
    try:
        return CollisionChecker(grid, vehicle, method, margin, n_circles, use_numba=use_numba)
    except TypeError:
        pytest.skip("CollisionChecker has no use_numba switch")


def grid_with_cells(cells, w=20.0, h=10.0, res=0.1):
    g = OccupancyGrid.empty(w, h, res)
    for ix, iy in cells:
        g.data[iy, ix] = 1
    return g


def footprint_polygon(vehicle, pose, margin=0.0) -> Polygon:
    return Polygon(vehicle.footprint_world(*pose, margin=margin))


def random_map(rng, w=20.0, h=20.0, res=0.1, n_boxes=8, density=0.0002):
    g = OccupancyGrid.empty(w, h, res)
    for _ in range(n_boxes):
        x0, y0 = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
        g.add_box(x0, y0, x0 + rng.uniform(0.1, 2.5), y0 + rng.uniform(0.1, 2.5))
    g.data[rng.random(g.data.shape) < density] = 1
    return g


def random_poses(rng, n, lo=3.0, hi=17.0):
    return np.column_stack([rng.uniform(lo, hi, n), rng.uniform(lo, hi, n), rng.uniform(-math.pi, math.pi, n)])


# ------------------------------------------------------------------ basics
@pytest.mark.parametrize("method", ["rectangle", "circles"])
@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_completely_free_map(vehicle, rng, method, use_numba):
    g = OccupancyGrid.empty(20.0, 20.0, 0.1)
    cc = make_checker(g, vehicle, method, margin=0.1, use_numba=use_numba)
    poses = random_poses(rng, 300, 5.0, 15.0)
    assert not cc.check_poses(poses).any()
    assert cc.check_pose(10.0, 10.0, 1.0) is False


@pytest.mark.parametrize("method", CollisionChecker.METHODS)
@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_completely_occupied_map(vehicle, rng, method, use_numba):
    g = OccupancyGrid(np.ones((200, 200)), 0.1)
    cc = make_checker(g, vehicle, method, margin=0.0, use_numba=use_numba)
    poses = random_poses(rng, 200, 5.0, 15.0)
    assert cc.check_poses(poses).all()


@pytest.mark.parametrize("method", ["rectangle", "circles"])
@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_vehicle_inside_large_obstacle(vehicle, method, use_numba):
    # footprint entirely inside an occupied blob: no boundary cell touches it
    g = OccupancyGrid.empty(30.0, 20.0, 0.1)
    g.data[20:180, 20:280] = 1
    cc = make_checker(g, vehicle, method, use_numba=use_numba)
    assert cc.check_pose(15.0, 10.0, 0.3)


# ------------------------------------------------------------ corner/side
@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_front_corner_collision(vehicle, use_numba):
    inside = make_checker(grid_with_cells([(87, 59)]), vehicle, use_numba=use_numba)
    assert inside.check_pose(*POSE), "occupied cell just inside the front-left corner not detected"
    for cell in [(88, 59), (87, 60), (88, 60)]:
        cc = make_checker(grid_with_cells([cell]), vehicle, use_numba=use_numba)
        assert not cc.check_pose(*POSE), f"cell {cell} just outside the front-left corner reported"
    # front-right corner
    assert make_checker(grid_with_cells([(87, 40)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    assert not make_checker(grid_with_cells([(88, 40)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    assert not make_checker(grid_with_cells([(87, 39)]), vehicle, use_numba=use_numba).check_pose(*POSE)


@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_rear_corner_collision(vehicle, use_numba):
    # rear-right corner (4.03, 4.08)
    assert make_checker(grid_with_cells([(40, 40)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    for cell in [(39, 40), (40, 39), (39, 39)]:
        assert not make_checker(grid_with_cells([cell]), vehicle, use_numba=use_numba).check_pose(*POSE), cell
    # rear-left corner (4.03, 5.98)
    assert make_checker(grid_with_cells([(40, 59)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    assert not make_checker(grid_with_cells([(39, 59)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    assert not make_checker(grid_with_cells([(40, 60)]), vehicle, use_numba=use_numba).check_pose(*POSE)


@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_side_collision(vehicle, use_numba):
    # middle of the left side (y = 5.98) and right side (y = 4.08)
    assert make_checker(grid_with_cells([(65, 59)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    assert not make_checker(grid_with_cells([(65, 60)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    assert make_checker(grid_with_cells([(65, 40)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    assert not make_checker(grid_with_cells([(65, 39)]), vehicle, use_numba=use_numba).check_pose(*POSE)
    # a whole wall along the side, 2 cm away -> free; touching -> collision
    g = OccupancyGrid.empty(20.0, 10.0, 0.1)
    g.data[60, 30:95] = 1
    assert not make_checker(g, vehicle, use_numba=use_numba).check_pose(*POSE)
    g.data[59, 30:95] = 1
    assert make_checker(g, vehicle, use_numba=use_numba).check_pose(*POSE)


@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_rotated_vehicle(vehicle, use_numba):
    pose = (10.03, 10.07, math.pi / 4)
    poly = footprint_polygon(vehicle, pose)
    minx, miny, maxx, maxy = poly.bounds
    res = 0.1
    outside_cell = inside_cell = None
    best = -1.0
    for ix in range(int(minx / res), int(maxx / res) + 1):
        for iy in range(int(miny / res), int(maxy / res) + 1):
            cb = box(ix * res, iy * res, (ix + 1) * res, (iy + 1) * res)
            if not box(minx, miny, maxx, maxy).contains(cb):
                continue
            d = poly.distance(cb)
            if d > best:
                best, outside_cell = d, (ix, iy)
    assert best > 0.5, "test setup: expected a cell inside the AABB far from the rotated rectangle"
    c = poly.centroid
    inside_cell = (int(c.x / res), int(c.y / res))
    # an obstacle strictly inside the AABB but outside the rotated rectangle is free
    g = grid_with_cells([outside_cell], 20.0, 20.0)
    assert not make_checker(g, vehicle, use_numba=use_numba).check_pose(*pose)
    # an obstacle inside the rotated rectangle collides
    g = grid_with_cells([inside_cell], 20.0, 20.0)
    assert make_checker(g, vehicle, use_numba=use_numba).check_pose(*pose)
    # a cell just outside the rotated front-left corner (10 cm along the diagonal)
    fl = vehicle.footprint_world(*pose)[1]
    d = np.array([math.cos(pose[2]) - math.sin(pose[2]), math.sin(pose[2]) + math.cos(pose[2])])
    p_out = fl + 0.12 * d / np.linalg.norm(d)
    cell = (int(p_out[0] / res), int(p_out[1] / res))
    cb = box(cell[0] * res, cell[1] * res, (cell[0] + 1) * res, (cell[1] + 1) * res)
    expected = poly.intersects(cb)
    g = grid_with_cells([cell], 20.0, 20.0)
    assert make_checker(g, vehicle, use_numba=use_numba).check_pose(*pose) == expected


def test_intermediate_primitive_collision(vehicle):
    """Endpoint poses of a primitive are free but one intermediate sample collides."""
    ps = primitives_for(vehicle, n_steer=5, length_classes=("short", "medium", "long"))
    start = (10.03, 10.07, 0.3)
    res = 0.1
    found = None
    for i, prim in sorted(enumerate(ps), key=lambda t: -abs(t[1].dyaw)):
        poses, _ = ps.transform_samples(*start, mask=np.arange(len(ps)) == i)
        ends = footprint_polygon(vehicle, start).union(footprint_polygon(vehicle, poses[-1])).buffer(0.03)
        mid = footprint_polygon(vehicle, poses[len(poses) // 2])
        diff = mid.difference(ends)
        if diff.is_empty:
            continue
        minx, miny, maxx, maxy = diff.bounds
        for ix in range(int(minx / res), int(maxx / res) + 1):
            for iy in range(int(miny / res), int(maxy / res) + 1):
                cb = box(ix * res, iy * res, (ix + 1) * res, (iy + 1) * res)
                if cb.intersection(mid).area > 1e-4 and not cb.intersects(ends):
                    found = (i, poses, (ix, iy))
                    break
            if found:
                break
        if found:
            break
    assert found, "test setup: no primitive with a sweep outside both end footprints"
    i, poses, cell = found
    g = grid_with_cells([cell], 20.0, 20.0)
    for use_numba in (None,):
        cc = make_checker(g, vehicle, use_numba=use_numba)
        assert not cc.check_pose(*start)
        assert not cc.check_pose(*poses[-1])
        hit = cc.check_poses(poses)
        assert hit.any(), "intermediate sample collision missed"
        assert not hit[0] and not hit[-1]


@pytest.mark.parametrize("method", ["rectangle", "circles"])
@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_map_boundary_collision(vehicle, method, use_numba):
    g = OccupancyGrid.empty(20.0, 10.0, 0.1)
    cc = make_checker(g, vehicle, method, margin=0.0, use_numba=use_numba)
    assert not cc.check_pose(1.05, 5.0, 0.0)      # rear at 0.05 -> inside
    assert cc.check_pose(0.95, 5.0, 0.0)          # rear at -0.05 -> outside
    assert cc.check_pose(17.0, 5.0, 0.0)          # front at 20.69
    assert not cc.check_pose(16.2, 5.0, 0.0)      # front at 19.89
    assert cc.check_pose(8.0, 9.5, 0.0)           # left side above the map
    assert cc.check_pose(8.0, 0.5, 0.0)           # right side below the map
    assert cc.check_pose(8.0, 8.5, math.pi / 2)   # front sticks out of the top
    assert cc.check_pose(-5.0, -5.0, 0.0)         # completely outside
    # margin pushes the footprint outside
    cm = make_checker(g, vehicle, method, margin=0.1, use_numba=use_numba)
    assert cm.check_pose(1.05, 5.0, 0.0)
    # with obstacles on the map (different code path)
    g2 = g.copy()
    g2.data[0, 0] = 1
    c2 = make_checker(g2, vehicle, method, margin=0.0, use_numba=use_numba)
    assert c2.check_pose(0.95, 5.0, 0.0)
    assert c2.check_pose(17.0, 5.0, 0.0)


@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_margin_effect(vehicle, use_numba):
    # cell (90, 50) spans x in [9.0, 9.1]: 0.28 m in front of the bumper (x = 8.72)
    g = grid_with_cells([(90, 50)])
    for margin, expected in ((0.0, False), (0.1, False), (0.25, False), (0.3, True), (0.5, True)):
        cc = make_checker(g, vehicle, margin=margin, use_numba=use_numba)
        assert cc.check_pose(*POSE) == expected, f"margin={margin}"
    # lateral: cell (65, 62) spans y in [6.2, 6.3]: 0.22 m left of y = 5.98
    g = grid_with_cells([(65, 62)])
    assert not make_checker(g, vehicle, margin=0.2, use_numba=use_numba).check_pose(*POSE)
    assert make_checker(g, vehicle, margin=0.24, use_numba=use_numba).check_pose(*POSE)


@pytest.mark.parametrize("n_circles", [2, 4, 8])
@pytest.mark.parametrize("margin", [0.0, 0.1])
@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_circles_method_is_conservative(vehicle, n_circles, margin, use_numba):
    rng = np.random.default_rng(7 + n_circles)
    g = random_map(rng)
    rect = make_checker(g, vehicle, "rectangle", margin, n_circles, use_numba=use_numba)
    circ = make_checker(g, vehicle, "circles", margin, n_circles, use_numba=use_numba)
    poses = random_poses(rng, 3000, 1.0, 19.0)
    r = rect.check_poses(poses)
    c = circ.check_poses(poses)
    assert r.sum() > 100 and (~r).sum() > 100, "test setup: need both outcomes"
    missed = np.nonzero(r & ~c)[0]
    assert len(missed) == 0, f"circles method missed rectangle collisions at poses {poses[missed[:5]]}"
    # it is an over-approximation, so it may reject more but not fewer poses
    assert c.sum() >= r.sum()


def _brute_force(vehicle, grid, poses, margin, tol=1e-3):
    """Shapely reference: +1 collision, 0 free, -1 ambiguous (within tol of a cell edge)."""
    iy, ix = np.nonzero(grid.data)
    r = grid.resolution
    ox, oy = grid.origin
    cells = shapely.box(ox + ix * r, oy + iy * r, ox + (ix + 1) * r, oy + (iy + 1) * r)
    tree = shapely.STRtree(cells)
    xmin, xmax, ymin, ymax = grid.extent
    out = np.empty(len(poses), dtype=int)
    for k, p in enumerate(poses):
        corners = vehicle.footprint_world(*p, margin=margin)
        if (np.any(corners[:, 0] < xmin - tol) or np.any(corners[:, 0] >= xmax + tol)
                or np.any(corners[:, 1] < ymin - tol) or np.any(corners[:, 1] >= ymax + tol)):
            out[k] = 1
            continue
        if (np.any(corners[:, 0] < xmin + tol) or np.any(corners[:, 0] > xmax - tol)
                or np.any(corners[:, 1] < ymin + tol) or np.any(corners[:, 1] > ymax - tol)):
            out[k] = -1
            continue
        poly = Polygon(corners)
        grown = poly.buffer(tol, join_style="mitre")
        if len(tree.query(grown, predicate="intersects")) == 0:
            out[k] = 0
            continue
        shrunk = poly.buffer(-tol, join_style="mitre")
        hits = tree.query(shrunk, predicate="intersects")
        # interior overlap (ignore touching within 1e-6)
        if any(cells[j].intersection(poly).area > 1e-6 * r for j in hits):
            out[k] = 1
        else:
            out[k] = -1
    return out


@pytest.mark.parametrize("margin", [0.0, 0.1, 0.25])
@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_rectangle_matches_shapely_brute_force(vehicle, margin, use_numba):
    rng = np.random.default_rng(int(margin * 100) + 3)
    g = random_map(rng, n_boxes=10, density=0.0003)
    cc = make_checker(g, vehicle, "rectangle", margin, use_numba=use_numba)
    poses = random_poses(rng, 800, 1.0, 19.0)
    ref = _brute_force(vehicle, g, poses, margin)
    got = cc.check_poses(poses)
    valid = ref >= 0
    assert (ref == 1).sum() > 100 and (ref == 0).sum() > 100, "test setup: need both outcomes"
    bad = np.nonzero(valid & (got != (ref == 1)))[0]
    assert len(bad) == 0, (f"{len(bad)} mismatches vs shapely (margin={margin}), e.g. poses "
                           f"{poses[bad[:3]].tolist()} ref={ref[bad[:3]].tolist()} got={got[bad[:3]].tolist()}")


@pytest.mark.parametrize("use_numba", NUMBA_MODES)
def test_rectangle_matches_shapely_near_obstacles(vehicle, use_numba):
    """Poses concentrated close to obstacle edges (where the exact test decides)."""
    rng = np.random.default_rng(99)
    g = random_map(rng, n_boxes=10, density=0.0)
    occ = g.occupied_centers()
    pick = occ[rng.integers(0, len(occ), 800)]
    yaw = rng.uniform(-math.pi, math.pi, 800)
    # place a random footprint point near the chosen occupied cell
    lx = rng.uniform(-2.5, 5.2, 800)
    ly = rng.uniform(-2.2, 2.2, 800)
    x = pick[:, 0] - (np.cos(yaw) * lx - np.sin(yaw) * ly)
    y = pick[:, 1] - (np.sin(yaw) * lx + np.cos(yaw) * ly)
    poses = np.column_stack([x, y, yaw])
    cc = make_checker(g, vehicle, "rectangle", 0.1, use_numba=use_numba)
    ref = _brute_force(vehicle, g, poses, 0.1)
    got = cc.check_poses(poses)
    valid = ref >= 0
    assert (ref == 0).sum() > 30
    bad = np.nonzero(valid & (got != (ref == 1)))[0]
    assert len(bad) == 0, f"mismatches vs shapely at {poses[bad[:3]].tolist()} ref={ref[bad[:3]].tolist()}"


def test_footprint_corners_match_vehicle(vehicle):
    g = OccupancyGrid.empty(10.0, 10.0, 0.1)
    for margin in (0.0, 0.2):
        cc = CollisionChecker(g, vehicle, "rectangle", margin, 4)
        poses = np.array([[3.0, 4.0, 0.0], [5.0, 5.0, 1.2], [2.0, 7.0, -2.9]])
        corners = cc.footprint_corners(poses)
        assert corners.shape == (3, 4, 2)
        for k, p in enumerate(poses):
            ref = vehicle.footprint_world(*p, margin=margin)
            a = sorted(map(tuple, np.round(corners[k], 9)))
            b = sorted(map(tuple, np.round(ref, 9)))
            np.testing.assert_allclose(a, b, atol=1e-9)


def test_point_method_semantics(vehicle):
    g = grid_with_cells([(87, 59), (60, 50)])
    cc = CollisionChecker(g, vehicle, "point", 0.0, 4)
    # footprint overlaps (87,59) but the rear axle point is free
    assert not cc.check_pose(*POSE)
    # rear axle on an occupied cell
    assert cc.check_pose(6.05, 5.05, 0.0)
    assert cc.check_pose(6.05, 5.05, 2.0)  # yaw irrelevant
    # outside the map counts as occupied, point near the edge is free
    assert cc.check_pose(-0.01, 5.0, 0.0)
    assert cc.check_pose(20.0, 5.0, 0.0)
    assert not cc.check_pose(0.01, 5.0, 0.0)
    out = cc.check_poses(np.array([[6.05, 5.05, 0.0], [1.0, 1.0, 0.0], [30.0, 1.0, 0.0]]))
    assert out.tolist() == [True, False, True]


@pytest.mark.parametrize("method", CollisionChecker.METHODS)
def test_check_count(vehicle, method):
    g = grid_with_cells([(50, 50)])
    cc = CollisionChecker(g, vehicle, method, 0.1, 4)
    assert cc.check_count == 0
    cc.check_pose(*POSE)
    assert cc.check_count == 1
    cc.check_poses(np.zeros((17, 3)) + [5.0, 5.0, 0.0])
    assert cc.check_count == 18
    cc.check_poses(np.zeros((0, 3)))
    assert cc.check_count == 18
    assert cc.check_poses(np.zeros((0, 3))).shape == (0,)


def test_min_clearance(vehicle):
    # obstacle column x in [9.0, 9.1]: 0.28 m in front of the bumper at x = 8.72
    g = OccupancyGrid.empty(20.0, 10.0, 0.1)
    g.data[45:56, 90] = 1
    cc = CollisionChecker(g, vehicle, "rectangle", 0.1, 4)
    d = cc.min_clearance(np.array([POSE]))[0]
    assert d == pytest.approx(0.28, abs=0.05)
    # moving 1 m back increases the clearance by ~1 m
    d2 = cc.min_clearance(np.array([[POSE[0] - 1.0, POSE[1], 0.0]]))[0]
    assert d2 == pytest.approx(1.28, abs=0.05)
    # capped value far away and on a free map
    assert cc.min_clearance(np.array([[POSE[0] - 4.0, POSE[1], 0.0]]), cap=1.0)[0] == pytest.approx(1.0)
    free = CollisionChecker(OccupancyGrid.empty(20.0, 10.0, 0.1), vehicle, "rectangle", 0.1, 4)
    assert free.min_clearance(np.array([POSE]), cap=3.0)[0] == pytest.approx(3.0)
    # lateral obstacle: row y in [6.5, 6.6] -> 0.52 m left of y = 5.98
    g = OccupancyGrid.empty(20.0, 10.0, 0.1)
    g.data[65, 40:80] = 1
    cc = CollisionChecker(g, vehicle, "rectangle", 0.1, 4)
    assert cc.min_clearance(np.array([POSE]))[0] == pytest.approx(0.52, abs=0.05)
    # the clearance estimate used in the cost is a rough (non-negative) value of the same order
    est = cc.clearance_estimate(np.array([POSE]))[0]
    assert 0.0 <= est <= 0.52 + 0.15


def test_invalid_arguments(vehicle):
    g = OccupancyGrid.empty(5.0, 5.0, 0.1)
    with pytest.raises(ValueError):
        CollisionChecker(g, vehicle, "triangle", 0.1, 4)
    with pytest.raises(ValueError):
        CollisionChecker(g, vehicle, "rectangle", -0.1, 4)
