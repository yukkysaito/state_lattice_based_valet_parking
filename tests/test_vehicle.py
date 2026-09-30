"""Tests for vehicle.py: parameters, footprint and kinematic bicycle model."""
import math

import numpy as np
import pytest

from vehicle import (InvalidVehicleParameterError, KinematicBicycleModel, VehicleInfo,
                     default_vehicle, wrap_angle)


def _model(v=None):
    return KinematicBicycleModel(v or default_vehicle())


def test_straight_forward_integration():
    m = _model()
    out = m.propagate(1.0, 2.0, 0.0, 0.0, +1, 5.0, ds=0.1)
    assert out.shape[1] == 3
    np.testing.assert_allclose(out[0], [1.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[-1], [6.0, 2.0, 0.0], atol=1e-9)
    # y and yaw stay constant, x monotonically increasing
    assert np.allclose(out[:, 1], 2.0) and np.allclose(out[:, 2], 0.0)
    assert np.all(np.diff(out[:, 0]) > 0)
    # heading 90deg
    out = m.propagate(0.0, 0.0, math.pi / 2, 0.0, +1, 3.0)
    np.testing.assert_allclose(out[-1], [0.0, 3.0, math.pi / 2], atol=1e-9)


def test_straight_reverse_integration():
    m = _model()
    out = m.propagate(1.0, 2.0, 0.0, 0.0, -1, 5.0, ds=0.1)
    np.testing.assert_allclose(out[-1], [-4.0, 2.0, 0.0], atol=1e-9)
    assert np.all(np.diff(out[:, 0]) < 0)
    out = m.propagate(0.0, 0.0, math.pi / 4, 0.0, -1, math.sqrt(2.0))
    np.testing.assert_allclose(out[-1], [-1.0, -1.0, math.pi / 4], atol=1e-9)


def test_constant_left_steering():
    v = default_vehicle()
    m = _model(v)
    steer = 0.3
    R = v.wheel_base / math.tan(steer)
    out = m.propagate(0.0, 0.0, 0.0, steer, +1, 4.0, ds=0.05)
    # yaw increases monotonically going forward with left steering
    assert np.all(np.diff(out[:, 2]) > 0)
    assert out[-1, 2] == pytest.approx(4.0 / R, abs=1e-9)
    # every sample lies on the circle centered at (0, R) with radius R
    r = np.hypot(out[:, 0] - 0.0, out[:, 1] - R)
    np.testing.assert_allclose(r, R, atol=1e-9)
    assert np.all(out[1:, 1] > 0)  # turns to the left
    # reverse with left steering: yaw decreases, still on the same circle
    back = m.propagate(0.0, 0.0, 0.0, steer, -1, 4.0)
    assert np.all(np.diff(back[:, 2]) < 0)
    np.testing.assert_allclose(np.hypot(back[:, 0], back[:, 1] - R), R, atol=1e-9)


def test_constant_right_steering():
    v = default_vehicle()
    m = _model(v)
    steer = -0.3
    R = v.wheel_base / math.tan(abs(steer))
    out = m.propagate(0.0, 0.0, 0.0, steer, +1, 4.0, ds=0.05)
    assert np.all(np.diff(out[:, 2]) < 0)
    assert out[-1, 2] == pytest.approx(-4.0 / R, abs=1e-9)
    np.testing.assert_allclose(np.hypot(out[:, 0], out[:, 1] + R), R, atol=1e-9)
    assert np.all(out[1:, 1] < 0)


def test_minimum_turning_radius():
    v = default_vehicle()
    assert v.min_turning_radius == pytest.approx(v.wheel_base / math.tan(v.max_steer_angle))
    assert v.max_curvature == pytest.approx(1.0 / v.min_turning_radius)
    v2 = VehicleInfo(wheel_base=3.0, max_steer_angle=math.radians(30))
    assert v2.min_turning_radius == pytest.approx(3.0 / math.tan(math.radians(30)))
    # a full circle at max steer returns to the start and has diameter 2R
    m = _model(v2)
    R = v2.min_turning_radius
    out = m.propagate(0, 0, 0, v2.max_steer_angle, 1, 2 * math.pi * R, ds=0.01)
    np.testing.assert_allclose(out[-1, :2], [0, 0], atol=1e-9)
    assert out[:, 1].max() == pytest.approx(2 * R, abs=1e-3)


def test_forward_reverse_symmetry():
    m = _model()
    for steer in (-0.5, -0.2, 0.0, 0.25, 0.55):
        start = (1.5, -2.0, 0.7)
        fwd = m.propagate(*start, steer, +1, 3.3)
        end = fwd[-1]
        back = m.propagate(end[0], end[1], end[2], steer, -1, 3.3)
        np.testing.assert_allclose(back[-1, :2], start[:2], atol=1e-9)
        assert abs(wrap_angle(back[-1, 2] - start[2])) < 1e-9
        # the reverse samples retrace the forward samples
        np.testing.assert_allclose(back[::-1, :2], fwd[:, :2], atol=1e-9)


def test_derived_length_width():
    v = VehicleInfo(wheel_base=2.5, wheel_tread=1.5, front_overhang=0.8, rear_overhang=0.6,
                    left_overhang=0.1, right_overhang=0.2, max_steer_angle=0.5)
    assert v.vehicle_length == pytest.approx(0.8 + 2.5 + 0.6)
    assert v.vehicle_width == pytest.approx(1.5 + 0.1 + 0.2)
    assert v.half_length == pytest.approx(v.vehicle_length / 2)
    assert v.half_width == pytest.approx(v.vehicle_width / 2)
    cx, cy = v.footprint_center_offset
    assert cx == pytest.approx((2.5 + 0.8 - 0.6) / 2)
    assert cy == pytest.approx((0.1 - 0.2) / 2)


def test_footprint_corners():
    v = VehicleInfo(wheel_base=2.5, wheel_tread=1.5, front_overhang=0.8, rear_overhang=0.6,
                    left_overhang=0.1, right_overhang=0.2, max_steer_angle=0.5)
    fl = v.footprint_local()
    front, rear = 2.5 + 0.8, -0.6
    left, right = 0.75 + 0.1, -(0.75 + 0.2)
    expected = {(front, right), (front, left), (rear, left), (rear, right)}
    got = {(round(a, 9), round(b, 9)) for a, b in fl}
    assert got == {(round(a, 9), round(b, 9)) for a, b in expected}
    # CCW orientation (positive signed area)
    x, y = fl[:, 0], fl[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area == pytest.approx(v.vehicle_length * v.vehicle_width)
    # margin inflates every side
    m = 0.2
    fm = v.footprint_local(margin=m)
    assert fm[:, 0].max() == pytest.approx(front + m) and fm[:, 0].min() == pytest.approx(rear - m)
    assert fm[:, 1].max() == pytest.approx(left + m) and fm[:, 1].min() == pytest.approx(right - m)
    # world transform: yaw = 90deg rotates forward (+x) onto +y
    fw = v.footprint_world(10.0, 5.0, math.pi / 2)
    np.testing.assert_allclose(fw, np.column_stack([10.0 - fl[:, 1], 5.0 + fl[:, 0]]), atol=1e-12)


def test_closed_form_vs_euler():
    m = _model()
    for steer, d in ((0.4, 1), (-0.3, 1), (0.5, -1), (0.0, -1)):
        length, ds = 3.0, 1e-4
        closed = m.propagate(0.3, -0.2, 0.4, steer, d, length)[-1]
        x, y, yaw = 0.3, -0.2, 0.4
        for _ in range(int(round(length / ds))):
            x, y, yaw = m.step_euler(x, y, yaw, steer, d, ds)
        assert math.hypot(x - closed[0], y - closed[1]) < 1e-3
        assert abs(wrap_angle(yaw - closed[2])) < 1e-6


@pytest.mark.parametrize("kwargs", [
    dict(wheel_base=0.0), dict(wheel_base=-1.0), dict(wheel_tread=0.0),
    dict(front_overhang=-0.1), dict(rear_overhang=-0.01), dict(left_overhang=-0.2),
    dict(right_overhang=-0.2), dict(wheel_base=float("nan")), dict(front_overhang=float("nan")),
    dict(wheel_base=float("inf")), dict(max_steer_angle=float("nan")), dict(max_steer_angle=0.0),
    dict(max_steer_angle=-0.3), dict(max_steer_angle=math.radians(80)), dict(max_steer_angle=2.0),
])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(InvalidVehicleParameterError):
        VehicleInfo(**kwargs)
    with pytest.raises(InvalidVehicleParameterError):
        default_vehicle().with_changes(**kwargs)


def test_invalid_parameter_error_is_value_error():
    assert issubclass(InvalidVehicleParameterError, ValueError)
    VehicleInfo(front_overhang=0.0, rear_overhang=0.0, left_overhang=0.0, right_overhang=0.0)
