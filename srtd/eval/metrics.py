from __future__ import annotations

import numpy as np
import torch

from srtd.spectral.residual import high_frequency_residual_energy


def path_length(path: np.ndarray) -> float:
    diffs = np.diff(path, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def average_squared_acceleration(path: np.ndarray, dt: float = 0.1) -> float:
    if len(path) < 3:
        return 0.0
    accel = (path[2:] - 2.0 * path[1:-1] + path[:-2]) / (dt * dt)
    return float(np.square(accel).sum(axis=1).mean())


def endpoint_error(path: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(path[-1] - goal))


def generated_high_freq_residual_energy(actions: np.ndarray, cutoff_norm: float = 0.2) -> float:
    tensor = torch.as_tensor(actions[None], dtype=torch.float32)
    return float(high_frequency_residual_energy(tensor, cutoff_norm=cutoff_norm).item())

