from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class RectObstacle:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def padded(self, padding: float) -> "RectObstacle":
        return RectObstacle(
            self.xmin - padding,
            self.ymin - padding,
            self.xmax + padding,
            self.ymax + padding,
        )

    def contains(self, point: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass
class MazeEnv:
    bounds: tuple[float, float, float, float]
    obstacles: list[RectObstacle]
    padding: float = 0.1
    _padded_obstacles: list[RectObstacle] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._padded_obstacles = [obs.padded(self.padding) for obs in self.obstacles]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MazeEnv":
        with Path(path).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(
            bounds=tuple(float(v) for v in cfg["bounds"]),
            obstacles=[RectObstacle(*map(float, obs)) for obs in cfg["obstacles"]],
            padding=float(cfg.get("padding", 0.1)),
        )

    @property
    def padded_obstacles(self) -> list[RectObstacle]:
        return self._padded_obstacles

    def in_bounds(self, point: np.ndarray) -> bool:
        xmin, ymin, xmax, ymax = self.bounds
        return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax

    def is_free(self, point: np.ndarray, padded: bool = True) -> bool:
        if not self.in_bounds(point):
            return False
        obstacles = self.padded_obstacles if padded else self.obstacles
        return not any(obs.contains(point) for obs in obstacles)

    def segment_is_free(
        self,
        a: np.ndarray,
        b: np.ndarray,
        step: float = 0.025,
        padded: bool = True,
    ) -> bool:
        dist = float(np.linalg.norm(b - a))
        n = max(2, int(np.ceil(dist / step)) + 1)
        for alpha in np.linspace(0.0, 1.0, n):
            if not self.is_free((1.0 - alpha) * a + alpha * b, padded=padded):
                return False
        return True

    def path_is_free(self, path: np.ndarray, padded: bool = True) -> bool:
        return all(
            self.segment_is_free(path[i], path[i + 1], padded=padded)
            for i in range(len(path) - 1)
        )

    def sample_free(self, rng: np.random.Generator) -> np.ndarray:
        xmin, ymin, xmax, ymax = self.bounds
        for _ in range(10_000):
            p = np.asarray([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
            if self.is_free(p):
                return p.astype(np.float32)
        raise RuntimeError("failed to sample a free point")
