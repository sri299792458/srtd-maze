from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from srtd.data.normalization import normalize_xy


@dataclass
class MazeEpisode:
    positions: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    source: str
    episode_id: int

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.start = np.asarray(self.start, dtype=np.float32)
        self.goal = np.asarray(self.goal, dtype=np.float32)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape [T, 2]")
        if self.source not in {"p", "q"}:
            raise ValueError("source must be 'p' for clean or 'q' for auxiliary")


@dataclass
class MazeChunk:
    obs: np.ndarray
    actions: np.ndarray
    source_id: int
    episode_id: int
    chunk_start: int
    sr_tmin_idx: int | None = None
    sr_bad_score: float | None = None


def build_chunks(
    episodes: Iterable[MazeEpisode],
    policy_horizon: int = 16,
    obs_horizon: int = 2,
    stride: int = 4,
    extent_m: float = 5.0,
) -> list[MazeChunk]:
    if obs_horizon != 2:
        raise ValueError("this prototype currently expects obs_horizon=2")

    chunks: list[MazeChunk] = []
    for episode in episodes:
        positions = episode.positions
        if len(positions) < obs_horizon + policy_horizon:
            continue
        goal_norm = normalize_xy(episode.goal, extent_m=extent_m)
        source_id = 0 if episode.source == "p" else 1
        last_start = len(positions) - policy_horizon
        for start_idx in range(obs_horizon - 1, last_start, stride):
            prev_xy = normalize_xy(positions[start_idx - 1], extent_m=extent_m)
            curr_xy = normalize_xy(positions[start_idx], extent_m=extent_m)
            actions = normalize_xy(
                positions[start_idx + 1 : start_idx + 1 + policy_horizon],
                extent_m=extent_m,
            )
            obs = np.concatenate([prev_xy, curr_xy, goal_norm]).astype(np.float32)
            chunks.append(
                MazeChunk(
                    obs=obs,
                    actions=actions.astype(np.float32),
                    source_id=source_id,
                    episode_id=episode.episode_id,
                    chunk_start=start_idx,
                )
            )
    return chunks


def chunks_to_arrays(chunks: list[MazeChunk]) -> dict[str, np.ndarray]:
    return {
        "obs": np.stack([c.obs for c in chunks]).astype(np.float32),
        "actions": np.stack([c.actions for c in chunks]).astype(np.float32),
        "source_id": np.asarray([c.source_id for c in chunks], dtype=np.int64),
        "episode_id": np.asarray([c.episode_id for c in chunks], dtype=np.int64),
        "chunk_start": np.asarray([c.chunk_start for c in chunks], dtype=np.int64),
        "sr_tmin_idx": np.asarray(
            [-1 if c.sr_tmin_idx is None else c.sr_tmin_idx for c in chunks],
            dtype=np.int64,
        ),
    }


def save_episodes(path: str | Path, episodes: list[MazeEpisode]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=np.asarray([e.positions for e in episodes], dtype=object),
        starts=np.asarray([e.start for e in episodes], dtype=np.float32),
        goals=np.asarray([e.goal for e in episodes], dtype=np.float32),
        sources=np.asarray([e.source for e in episodes]),
        episode_ids=np.asarray([e.episode_id for e in episodes], dtype=np.int64),
    )


def load_episodes(path: str | Path) -> list[MazeEpisode]:
    data = np.load(path, allow_pickle=True)
    return [
        MazeEpisode(
            positions=np.asarray(pos, dtype=np.float32),
            start=data["starts"][i],
            goal=data["goals"][i],
            source=str(data["sources"][i]),
            episode_id=int(data["episode_ids"][i]),
        )
        for i, pos in enumerate(data["positions"])
    ]

