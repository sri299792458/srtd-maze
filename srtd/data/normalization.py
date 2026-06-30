from __future__ import annotations

import numpy as np


def normalize_xy(xy_meters: np.ndarray, extent_m: float = 5.0) -> np.ndarray:
    return (2.0 * xy_meters / extent_m - 1.0).astype(np.float32)


def denormalize_xy(xy_norm: np.ndarray, extent_m: float = 5.0) -> np.ndarray:
    return (0.5 * extent_m * (xy_norm + 1.0)).astype(np.float32)

