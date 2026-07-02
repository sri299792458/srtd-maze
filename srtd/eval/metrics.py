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


def _fixed_windows(values: np.ndarray, window: int, stride: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros((1, window, 2), dtype=np.float32)
    if len(values) < window:
        pad = np.repeat(values[-1:], window - len(values), axis=0)
        return np.concatenate([values, pad], axis=0)[None]
    starts = range(0, len(values) - window + 1, stride)
    return np.stack([values[i : i + window] for i in starts]).astype(np.float32)


def generated_high_freq_residual_energy_absolute(
    path: np.ndarray,
    cutoff_norm: float = 0.2,
    window: int = 16,
    stride: int = 4,
) -> float:
    """High-frequency residual energy on fixed windows of absolute positions."""
    windows = _fixed_windows(path, window=window, stride=stride)
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    return float(high_frequency_residual_energy(tensor, cutoff_norm=cutoff_norm).mean().item())


def generated_high_freq_residual_energy_delta(
    path: np.ndarray,
    cutoff_norm: float = 0.2,
    window: int = 16,
    stride: int = 4,
) -> float:
    """High-frequency residual energy on fixed windows of delta motion."""
    path = np.asarray(path, dtype=np.float32)
    if len(path) < 2:
        return 0.0
    windows = _fixed_windows(np.diff(path, axis=0), window=window, stride=stride)
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    return float(high_frequency_residual_energy(tensor, cutoff_norm=cutoff_norm).mean().item())


def generated_high_freq_residual_energy(
    actions: np.ndarray,
    cutoff_norm: float = 0.2,
    window: int = 16,
    stride: int = 4,
) -> float:
    return generated_high_freq_residual_energy_delta(
        actions,
        cutoff_norm=cutoff_norm,
        window=window,
        stride=stride,
    )
