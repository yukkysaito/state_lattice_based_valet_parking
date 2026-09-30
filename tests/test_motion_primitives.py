"""Tests for motion_primitives.py."""
import math

import numpy as np
import pytest

from motion_primitives import (FORWARD, LENGTH_CLASS_FACTORS, REVERSE, PrimitiveConfig,
                               generate_primitives, primitives_for)
from vehicle import KinematicBicycleModel, VehicleInfo, default_vehicle, small_car


def _expected_count(v: VehicleInfo, cfg: PrimitiveConfig) -> int:
    """Independent count: 2 dirs * n_steer * n_classes minus (steer, length) duplicates."""
    if cfg.n_steer == 1:
        steers = [0.0]
    else:
        steers = list(np.linspace(-v.max_steer_angle, v.max_steer_angle, cfg.n_steer))
        steers[cfg.n_steer // 2] = 0.0
    min_len = 3.0 * cfg.xy_resolution
    uniq = set()
    for st in steers:
        k = math.tan(st) / v.wheel_base
        for cls in cfg.length_classes:
            target = max(min_len, LENGTH_CLASS_FACTORS[cls] * v.min_turning_radius)
            if abs(k) < 1e-9:
                L = target
            else:
                # smallest multiple of the yaw resolution close to the target and >= min_len
                n = max(1, int(round(abs(k) * target / cfg.yaw_resolution)))
                while n * cfg.yaw_resolution / abs(k) < min_len:
                    n += 1
                L = n * cfg.yaw_resolution / abs(k)
            uniq.add((round(st, 9), round(L, 6)))
    n_dirs = 2 if cfg.allow_reverse else 1
    return n_dirs * len(uniq)


@pytest.mark.parametrize("n_steer", [3, 5, 7])
@pytest.mark.parametrize("classes", [("short",), ("short", "medium"), ("short", "medium", "long")])
def test_primitive_count(vehicle, n_steer, classes):
    cfg = PrimitiveConfig(n_steer=n_steer, length_classes=classes)
    ps = generate_primitives(vehicle, cfg)
    assert len(ps) == _expected_count(vehicle, cfg)
    assert len(ps) <= 2 * n_steer * len(classes)
    assert len(ps) >= 2 * n_steer  # every steering value present in both directions
    # no duplicates
    keys = {(p.direction, round(p.steering, 9), round(p.length, 6)) for p in ps}
    assert len(keys) == len(ps)
    # every steering value appears
    assert len(ps.steering_values()) == n_steer
    # indices are consecutive
    assert [p.index for p in ps] == list(range(len(ps)))


def test_primitive_count_without_duplicates(vehicle):
    # for the default vehicle, 5 steer x (short, medium) has no length collision
    ps = primitives_for(vehicle, n_steer=5, length_classes=("short", "medium"))
    assert len(ps) == 2 * 5 * 2


@pytest.mark.parametrize("n_steer", [3, 5, 7])
def test_left_right_symmetry(vehicle, n_steer):
    ps = primitives_for(vehicle, n_steer=n_steer, length_classes=("short", "medium", "long"))
    for p in ps:
        mirror = [q for q in ps if q.direction == p.direction and abs(q.steering + p.steering) < 1e-9
                  and abs(q.length - p.length) < 1e-9]
        assert len(mirror) == 1, f"primitive {p.index} has no unique mirror"
        q = mirror[0]
        assert q.dx == pytest.approx(p.dx, abs=1e-9)
        assert q.dy == pytest.approx(-p.dy, abs=1e-9)
        assert q.dyaw == pytest.approx(-p.dyaw, abs=1e-9)
        np.testing.assert_allclose(q.samples[:, 1], -p.samples[:, 1], atol=1e-9)


def test_forward_reverse_generation(vehicle):
    ps = primitives_for(vehicle, n_steer=5)
    fwd = [p for p in ps if p.direction == FORWARD]
    rev = [p for p in ps if p.direction == REVERSE]
    assert fwd and rev and len(fwd) == len(rev)
    assert all(p.dx > 0 for p in fwd)
    assert all(p.dx < 0 for p in rev)
    # reverse primitives are the time-mirror (x -> -x, yaw -> -yaw) of the forward ones
    for p in fwd:
        q = [r for r in rev if abs(r.steering - p.steering) < 1e-12 and abs(r.length - p.length) < 1e-12]
        assert len(q) == 1
        assert q[0].dx == pytest.approx(-p.dx) and q[0].dy == pytest.approx(p.dy)
        assert q[0].dyaw == pytest.approx(-p.dyaw)
    # reverse + left steer turns the heading clockwise
    for p in rev:
        if p.steering > 0:
            assert p.dyaw < 0


def test_allow_reverse_false(vehicle):
    ps = primitives_for(vehicle, n_steer=5, allow_reverse=False)
    assert len(ps) == 5 * 2
    assert all(p.direction == FORWARD for p in ps)
    assert np.all(ps.directions == FORWARD)


@pytest.mark.parametrize("v", [default_vehicle(), small_car()])
def test_no_primitive_exceeds_max_steer(v):
    ps = primitives_for(v, n_steer=7, length_classes=("short", "medium", "long"))
    assert np.all(np.abs(ps.steerings) <= v.max_steer_angle + 1e-12)
    assert np.isclose(np.abs(ps.steerings).max(), v.max_steer_angle)
    for p in ps:
        assert abs(p.curvature) <= v.max_curvature + 1e-12
        assert p.curvature == pytest.approx(math.tan(p.steering) / v.wheel_base)
        # the sampled geometry has curvature |dyaw/ds| <= kappa_max
        assert abs(p.dyaw) / p.length <= v.max_curvature + 1e-9


def test_endpoint_consistency(vehicle):
    ps = primitives_for(vehicle, n_steer=5, length_classes=("short", "medium", "long"))
    model = KinematicBicycleModel(vehicle)
    for i, p in enumerate(ps):
        np.testing.assert_allclose(p.samples[-1], [p.dx, p.dy, p.dyaw], atol=1e-12)
        np.testing.assert_allclose(ps.ends[i], [p.dx, p.dy, p.dyaw], atol=1e-12)
        ref = model.propagate(0.0, 0.0, 0.0, p.steering, p.direction, p.length)[-1]
        np.testing.assert_allclose(ref, [p.dx, p.dy, p.dyaw], atol=1e-9)
        # samples exclude the start pose
        assert np.hypot(*p.samples[0, :2]) > 1e-9
    # transform_ends consistent with transform_samples of the last sample
    x, y, yaw = 3.0, -1.0, 2.5
    ends = ps.transform_ends(x, y, yaw)
    wposes, owner = ps.transform_samples(x, y, yaw)
    for i in range(len(ps)):
        last = wposes[owner == i][-1]
        np.testing.assert_allclose(last, ends[i], atol=1e-9)


@pytest.mark.parametrize("yaw_deg", [5.0, 3.0, 7.5])
def test_heading_lattice_consistency(vehicle, yaw_deg):
    res = math.radians(yaw_deg)
    ps = primitives_for(vehicle, n_steer=5, length_classes=("short", "medium", "long"), yaw_resolution=res)
    for p in ps:
        k = p.dyaw / res
        assert abs(k - round(k)) < 1e-6, f"dyaw={p.dyaw} is not a multiple of {res}"
        if p.steering != 0.0:
            assert round(abs(k)) >= 1
        else:
            assert p.dyaw == 0.0


@pytest.mark.parametrize("ds", [0.02, 0.05, 0.1])
def test_sample_spacing(vehicle, ds):
    ps = primitives_for(vehicle, n_steer=5, length_classes=("short", "medium", "long"), sample_ds=ds)
    for p in ps:
        pts = np.vstack([[0.0, 0.0], p.samples[:, :2]])
        step = np.hypot(*np.diff(pts, axis=0).T)
        assert np.all(step <= ds + 1e-9)
        assert len(p.samples) >= 2
        # arc length of samples adds up to the primitive length (chord <= arc)
        assert step.sum() <= p.length + 1e-9
        assert step.sum() == pytest.approx(p.length, rel=1e-3)


def test_primitives_change_with_vehicle_parameters(vehicle):
    cfg = PrimitiveConfig()
    base = generate_primitives(vehicle, cfg)
    longer = generate_primitives(vehicle.with_changes(wheel_base=3.4), cfg)
    lower = generate_primitives(vehicle.with_changes(max_steer_angle=0.4), cfg)
    assert base is not longer and base is not lower
    assert not np.allclose(base.lengths, longer.lengths)
    assert not np.allclose(base.ends, lower.ends)
    assert np.abs(lower.steerings).max() == pytest.approx(0.4)
    # a larger turning radius gives longer primitives (length classes scale with R_min)
    assert longer.lengths.max() > base.lengths.max()


def test_primitive_cache(vehicle):
    cfg = PrimitiveConfig(n_steer=5)
    a = generate_primitives(vehicle, cfg)
    b = generate_primitives(vehicle, PrimitiveConfig(n_steer=5))
    # an equal VehicleInfo instance (name is not part of the identity) hits the cache too
    c = generate_primitives(VehicleInfo(name="other_name"), PrimitiveConfig(n_steer=5))
    assert a is b
    assert vehicle == VehicleInfo(name="other_name")
    assert a is c
    d = generate_primitives(vehicle, PrimitiveConfig(n_steer=3))
    assert d is not a


def test_invalid_config_rejected(vehicle):
    for cfg in (PrimitiveConfig(n_steer=4), PrimitiveConfig(n_steer=0),
                PrimitiveConfig(length_classes=("huge",)), PrimitiveConfig(sample_ds=1.0)):
        with pytest.raises(ValueError):
            generate_primitives(vehicle, cfg)
