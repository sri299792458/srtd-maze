import numpy as np

from srtd.eval.metrics import average_squared_acceleration


def test_cubic_spline_smoothness_zero_for_line():
    t = np.arange(20, dtype=np.float64) * 0.1
    path = np.stack([0.5 * t, -0.25 * t], axis=-1)
    assert average_squared_acceleration(path, dt=0.1) < 1e-10


def test_cubic_spline_smoothness_matches_quadratic_acceleration():
    dt = 0.1
    t = np.arange(30, dtype=np.float64) * dt
    accel = np.asarray([2.0, -1.0])
    path = 0.5 * accel[None, :] * t[:, None] ** 2
    expected = float(np.square(accel).sum())
    assert abs(average_squared_acceleration(path, dt=dt) - expected) < 1e-3

