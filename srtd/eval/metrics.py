from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from srtd.spectral.residual import high_frequency_residual_energy


def path_length(path: np.ndarray) -> float:
    diffs = np.diff(path, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def finite_difference_squared_acceleration(path: np.ndarray, dt: float = 0.1) -> float:
    if len(path) < 3:
        return 0.0
    accel = (path[2:] - 2.0 * path[1:-1] + path[:-2]) / (dt * dt)
    return float(np.square(accel).sum(axis=1).mean())


def finite_difference_squared_jerk(path: np.ndarray, dt: float = 0.1) -> float:
    if len(path) < 4:
        return 0.0
    jerk = (path[3:] - 3.0 * path[2:-1] + 3.0 * path[1:-2] - path[:-3]) / (dt**3)
    return float(np.square(jerk).sum(axis=1).mean())


def mean_abs_turn_rate(path: np.ndarray, dt: float = 0.1) -> float:
    if len(path) < 3:
        return 0.0
    deltas = np.diff(path, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = (norms[:-1] > 1e-8) & (norms[1:] > 1e-8)
    if not valid.any():
        return 0.0
    unit_a = deltas[:-1][valid] / norms[:-1][valid, None]
    unit_b = deltas[1:][valid] / norms[1:][valid, None]
    cos = np.sum(unit_a * unit_b, axis=1).clip(-1.0, 1.0)
    angles = np.arccos(cos)
    return float(np.mean(np.abs(angles) / dt))


def mean_target_jump(targets: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.float32)
    if len(targets) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(targets, axis=0), axis=1).mean())


def cubic_spline_squared_acceleration(path: np.ndarray, dt: float = 0.1) -> float:
    """Mean integrated squared acceleration from a cubic spline."""
    if len(path) < 3:
        return 0.0
    path = np.asarray(path, dtype=np.float64)
    duration = dt * (len(path) - 1)
    if duration <= 0:
        return 0.0
    t = np.arange(len(path), dtype=np.float64) * dt
    spline = CubicSpline(t, path, axis=0)
    dense_count = max(200, 20 * len(path))
    dense_t = np.linspace(t[0], t[-1], dense_count)
    accel = spline(dense_t, 2)
    squared = np.square(accel).sum(axis=1)
    return float(np.trapezoid(squared, dense_t) / duration)


def average_squared_acceleration(path: np.ndarray, dt: float = 0.1) -> float:
    return cubic_spline_squared_acceleration(path, dt=dt)


def endpoint_error(path: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(path[-1] - goal))


def generated_high_freq_residual_energy(
    actions: np.ndarray,
    cutoff_norm: float = 0.2,
    window: int = 16,
    stride: int = 4,
) -> float:
    """High-frequency residual energy on fixed windows of delta motion."""
    actions = np.asarray(actions, dtype=np.float32)
    if len(actions) < 2:
        return 0.0
    deltas = np.diff(actions, axis=0)
    if len(deltas) < window:
        pad = np.repeat(deltas[-1:], window - len(deltas), axis=0) if len(deltas) else np.zeros((window, actions.shape[1]), dtype=np.float32)
        windows = np.concatenate([deltas, pad], axis=0)[None]
    else:
        starts = range(0, len(deltas) - window + 1, stride)
        windows = np.stack([deltas[i : i + window] for i in starts]).astype(np.float32)
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    return float(high_frequency_residual_energy(tensor, cutoff_norm=cutoff_norm).mean().item())
