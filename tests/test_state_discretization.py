"""Tests for state_lattice.StateDiscretizer and angle wrapping."""
import math

import numpy as np
import pytest

from state_lattice import StateDiscretizer
from vehicle import normalize_angle, wrap_angle

RES = 0.1
YAW = math.radians(5.0)


def test_xy_index_floor_semantics():
    d = StateDiscretizer(RES, YAW)
    assert d.xy_index(0.0, 0.0) == (0, 0)
    assert d.xy_index(0.05, 0.099) == (0, 0)
    assert d.xy_index(0.1000001, 0.25) == (1, 2)
    assert d.xy_index(3.95, 7.01) == (39, 70)
    # floor (not truncation) for negative coordinates
    assert d.xy_index(-0.01, -0.0001) == (-1, -1)
    assert d.xy_index(-0.15, -0.2000001) == (-2, -3)
    # cell centre maps back to the same index
    for ix, iy in [(0, 0), (5, -3), (-7, 12)]:
        cx, cy = d.xy_center(ix, iy)
        assert d.xy_index(cx, cy) == (ix, iy)


def test_xy_index_origin_offset():
    d = StateDiscretizer(RES, YAW, origin=(-5.0, 2.5))
    assert d.xy_index(-5.0, 2.5) == (0, 0)
    assert d.xy_index(-4.95, 2.55) == (0, 0)
    assert d.xy_index(-5.01, 2.49) == (-1, -1)
    assert d.xy_index(0.05, 3.05) == (50, 5)
    assert d.xy_center(0, 0) == pytest.approx((-4.95, 2.55))


def test_vectorised_keys_match_scalar(rng):
    d = StateDiscretizer(RES, YAW, origin=(-3.0, 1.0))
    poses = np.column_stack([rng.uniform(-10, 10, 500), rng.uniform(-10, 10, 500),
                             rng.uniform(-4, 4, 500)])
    dirs = rng.choice([-1, 1], 500)
    keys = d.keys(poses, dirs)
    for p, dr, k in zip(poses, dirs, keys):
        assert k == d.key(p[0], p[1], p[2], int(dr))


def test_yaw_wrapping(rng):
    d = StateDiscretizer(RES, YAW)
    assert d.n_yaw == 72
    yaws = rng.uniform(-3 * math.pi, 3 * math.pi, 500)
    # stay away from exact half-bin ties (rounding direction there is arbitrary)
    frac = np.abs((yaws / YAW) % 1.0 - 0.5)
    yaws = np.concatenate([yaws[frac > 1e-6], np.arange(-72, 73) * YAW])
    for yaw in yaws:
        i = d.yaw_index(yaw)
        assert 0 <= i < d.n_yaw
        assert d.yaw_index(yaw + 2 * math.pi) == i
        assert d.yaw_index(yaw - 2 * math.pi) == i
        # yaw_value returns the lattice heading of the bin (within half a bin of yaw)
        assert abs(wrap_angle(d.yaw_value(i) - yaw)) <= YAW / 2 + 1e-9
    assert d.yaw_index(0.0) == 0
    assert d.yaw_index(YAW) == 1
    assert d.yaw_index(-YAW) == d.n_yaw - 1


def test_pi_boundary_same_index():
    d = StateDiscretizer(RES, YAW)
    assert d.yaw_index(math.pi) == d.yaw_index(-math.pi)
    assert d.key(1.0, 1.0, math.pi, 1) == d.key(1.0, 1.0, -math.pi, 1)
    eps = 1e-9
    assert d.yaw_index(math.pi - eps) == d.yaw_index(-math.pi + eps)
    keys = d.keys(np.array([[1.0, 1.0, math.pi], [1.0, 1.0, -math.pi]]), np.array([1, 1]))
    assert keys[0] == keys[1]
    # also for a resolution that does not divide pi evenly into an odd count
    d2 = StateDiscretizer(RES, 2 * math.pi / 36)
    assert d2.yaw_index(math.pi) == d2.yaw_index(-math.pi)


def test_nearby_states_same_key():
    d = StateDiscretizer(RES, YAW)
    base = d.key(2.03, 4.07, 10 * YAW, 1)
    for dx, dy, dyaw in [(0.0, 0.0, 0.0), (0.06, 0.02, 0.49 * YAW), (-0.029, -0.069, -0.49 * YAW),
                         (0.069, 0.029, 0.3 * YAW)]:
        assert d.key(2.03 + dx, 4.07 + dy, 10 * YAW + dyaw, 1) == base
    # crossing the cell or the half-bin boundary changes the key
    assert d.key(2.101, 4.07, 10 * YAW, 1) != base
    assert d.key(2.03, 4.07, 10.51 * YAW, 1) != base
    assert d.key(2.03, 4.07, 9.49 * YAW, 1) != base


def test_direction_in_key():
    d = StateDiscretizer(RES, YAW, include_direction=True)
    assert d.key(1.0, 2.0, 0.3, 1) != d.key(1.0, 2.0, 0.3, -1)
    ks = d.keys(np.array([[1.0, 2.0, 0.3]] * 2), np.array([1, -1]))
    assert ks[0] != ks[1]
    d2 = StateDiscretizer(RES, YAW, include_direction=False)
    assert d2.key(1.0, 2.0, 0.3, 1) == d2.key(1.0, 2.0, 0.3, -1)
    ks = d2.keys(np.array([[1.0, 2.0, 0.3]] * 2), np.array([1, -1]))
    assert ks[0] == ks[1]


def test_invalid_resolution():
    with pytest.raises(ValueError):
        StateDiscretizer(0.0, YAW)
    with pytest.raises(ValueError):
        StateDiscretizer(RES, -1.0)


def test_normalize_and_wrap_angle_range(rng):
    a = np.concatenate([rng.uniform(-50, 50, 2000),
                        np.array([0.0, math.pi, -math.pi, 2 * math.pi, -2 * math.pi, 3 * math.pi,
                                  math.pi - 1e-12, -math.pi + 1e-12])])
    n = normalize_angle(a)
    assert np.all(n >= -math.pi) and np.all(n < math.pi)
    # same angle (difference is a multiple of 2 pi)
    k = (a - n) / (2 * math.pi)
    np.testing.assert_allclose(k, np.round(k), atol=1e-9)
    for x in a:
        w = wrap_angle(float(x))
        assert -math.pi <= w < math.pi
        assert w == pytest.approx(float(normalize_angle(x)), abs=1e-12)
    assert wrap_angle(math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert normalize_angle(np.array([math.pi]))[0] == pytest.approx(-math.pi)


def test_wrap_angle_float_edge_below_minus_pi():
    x = float(np.nextafter(-math.pi, -10.0))   # -3.1415926535897936
    w = wrap_angle(x)
    n = float(normalize_angle(x))
    assert -math.pi <= w < math.pi, f"wrap_angle({x!r}) = {w!r} is outside [-pi, pi)"
    assert -math.pi <= n < math.pi, f"normalize_angle({x!r}) = {n!r} is outside [-pi, pi)"
