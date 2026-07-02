import numpy as np

from srtd.eval.maze_env import MazeEnv, RectObstacle
from srtd.eval.metrics import average_squared_acceleration, finite_difference_squared_jerk, mean_abs_turn_rate, mean_target_jump


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


def test_stepwise_quality_metrics_detect_jumps_and_turns():
    path = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    assert finite_difference_squared_jerk(path, dt=1.0) > 0.0
    assert mean_abs_turn_rate(path, dt=1.0) > 0.0
    assert mean_target_jump(path) > 0.0


def test_maze_clearance_uses_padding():
    env = MazeEnv(bounds=(0.0, 0.0, 2.0, 2.0), obstacles=[RectObstacle(1.0, 0.5, 1.2, 1.5)], padding=0.1)
    point = np.asarray([0.95, 1.0], dtype=np.float32)
    assert env.obstacle_clearance(point, padded=False) > 0.0
    assert env.obstacle_clearance(point, padded=True) < 0.0
