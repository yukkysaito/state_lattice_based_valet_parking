"""Tests for reeds_shepp.py."""
import math

import numpy as np
import pytest

from reeds_shepp import rs_length_vec, rs_paths, rs_shortest_length, sample_rs_path
from vehicle import wrap_angle

R = 4.55


def _random_goals(rng, n, span=12.0):
    return np.column_stack([rng.uniform(-span, span, n), rng.uniform(-span, span, n),
                            rng.uniform(-math.pi, math.pi, n)])


def test_all_candidates_reach_goal(rng):
    starts = _random_goals(rng, 60, 5.0)
    goals = _random_goals(rng, 60, 10.0)
    n_checked = 0
    for s, g in zip(starts, goals):
        paths = rs_paths(tuple(s), tuple(g), R)
        assert paths, f"no RS path from {s} to {g}"
        lengths = [p.total_length for p in paths]
        assert lengths == sorted(lengths)
        for p in paths:
            poses, steer, dirs, seg = sample_rs_path(p, tuple(s), ds=0.05)
            assert len(poses) > 0
            end = poses[-1]
            assert math.hypot(end[0] - g[0], end[1] - g[1]) < 1e-3, f"{p.types} misses goal"
            assert abs(wrap_angle(end[2] - g[2])) < 1e-3
            assert np.all(np.abs(steer) <= 1) and set(np.unique(dirs)) <= {-1, 1}
            # sampled arc length equals the path length
            step = np.hypot(*np.diff(np.vstack([s[:2], poses[:, :2]]), axis=0).T)
            assert step.max() <= 0.05 + 1e-9
            assert step.sum() == pytest.approx(p.total_length, rel=1e-3, abs=1e-3)
            n_checked += 1
    assert n_checked > 200


def test_vectorised_length_matches_scalar(rng):
    goals = _random_goals(rng, 400, 15.0)
    vec = rs_length_vec(goals[:, 0], goals[:, 1], goals[:, 2], R)
    for g, v in zip(goals, vec):
        s = rs_shortest_length((0.0, 0.0, 0.0), tuple(g), R)
        assert v == pytest.approx(s, rel=1e-6, abs=1e-6), f"goal={tuple(g)}"


def test_vectorised_length_unit_radius_scaling(rng):
    goals = _random_goals(rng, 100, 10.0)
    a = rs_length_vec(goals[:, 0], goals[:, 1], goals[:, 2], R)
    b = rs_length_vec(goals[:, 0] / R, goals[:, 1] / R, goals[:, 2], 1.0) * R
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_straight_forward_backward():
    for d in (0.5, 3.0, 10.0):
        paths = rs_paths((0, 0, 0), (d, 0, 0), R)
        assert paths[0].total_length == pytest.approx(d)
        assert all(l >= 0 for _, l in paths[0].segments())
        assert paths[0].n_direction_changes == 0
        paths = rs_paths((0, 0, 0), (-d, 0, 0), R)
        assert paths[0].total_length == pytest.approx(d)
        assert all(l <= 0 for _, l in paths[0].segments())
        assert rs_shortest_length((0, 0, 0), (-d, 0, 0), R) == pytest.approx(d)
        np.testing.assert_allclose(rs_length_vec(np.array([d, -d]), np.zeros(2), np.zeros(2), R), [d, d])
    # rotated start frame
    s = (1.0, 2.0, 0.7)
    g = (1.0 + 4 * math.cos(0.7), 2.0 + 4 * math.sin(0.7), 0.7)
    assert rs_shortest_length(s, g, R) == pytest.approx(4.0)
    assert rs_shortest_length(s, s, R) == pytest.approx(0.0, abs=1e-9)


def test_symmetry_properties(rng):
    goals = _random_goals(rng, 300, 12.0)
    x, y, phi = goals.T
    base = rs_length_vec(x, y, phi, R)
    # timeflip: (x, y, phi) -> (-x, y, -phi)
    np.testing.assert_allclose(rs_length_vec(-x, y, -phi, R), base, rtol=1e-6, atol=1e-6)
    # reflect: (x, y, phi) -> (x, -y, -phi)
    np.testing.assert_allclose(rs_length_vec(x, -y, -phi, R), base, rtol=1e-6, atol=1e-6)
    # backwards: goal->start has the same length
    for g, b in zip(goals[:60], base[:60]):
        assert rs_shortest_length(tuple(g), (0.0, 0.0, 0.0), R) == pytest.approx(b, rel=1e-6, abs=1e-6)
    # lower bounds
    assert np.all(base >= np.hypot(x, y) - 1e-9)
    assert np.all(base >= R * np.abs(phi) - 1e-9)


def test_metric_triangle_inequality(rng):
    pts = _random_goals(rng, 40, 8.0)
    for a, b, c in zip(pts[:-2], pts[1:-1], pts[2:]):
        ab = rs_shortest_length(tuple(a), tuple(b), R)
        bc = rs_shortest_length(tuple(b), tuple(c), R)
        ac = rs_shortest_length(tuple(a), tuple(c), R)
        assert ac <= ab + bc + 1e-6
