from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from srtd.data.maze2d_dataset import MazeChunk
from srtd.diffusion.schedules import VPSchedule


@dataclass
class SampleBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    source_id: torch.Tensor
    t_idx: torch.Tensor
    indices: np.ndarray


class ChunkSampler:
    def __init__(
        self,
        chunks: list[MazeChunk],
        schedule: VPSchedule,
        source_weights: dict[int, float] | None = None,
        seed: int = 0,
    ) -> None:
        self.chunks = chunks
        self.schedule = schedule
        self.rng = np.random.default_rng(seed)
        self.source_to_indices = {
            source: np.asarray([i for i, c in enumerate(chunks) if c.source_id == source], dtype=np.int64)
            for source in (0, 1)
        }
        self.source_weights = source_weights or self._default_weights()

    def _default_weights(self) -> dict[int, float]:
        counts = {k: len(v) for k, v in self.source_to_indices.items() if len(v) > 0}
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def admissible(self, chunk: MazeChunk, t_idx: int) -> bool:
        return True

    def _sample_one_index(self, t_idx: int) -> int:
        sources = np.asarray(list(self.source_weights.keys()), dtype=np.int64)
        probs = np.asarray([self.source_weights[int(s)] for s in sources], dtype=np.float64)
        probs = probs / probs.sum()
        for _ in range(10_000):
            source = int(self.rng.choice(sources, p=probs))
            idx = int(self.rng.choice(self.source_to_indices[source]))
            if self.admissible(self.chunks[idx], t_idx):
                return idx
        raise RuntimeError("failed to draw an admissible chunk")

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> SampleBatch:
        t_idx = self.rng.integers(0, self.schedule.train_steps, size=batch_size, endpoint=False)
        indices = np.asarray([self._sample_one_index(int(t)) for t in t_idx], dtype=np.int64)
        obs = np.stack([self.chunks[i].obs for i in indices]).astype(np.float32)
        actions = np.stack([self.chunks[i].actions for i in indices]).astype(np.float32)
        source = np.asarray([self.chunks[i].source_id for i in indices], dtype=np.int64)
        return SampleBatch(
            obs=torch.as_tensor(obs, device=device),
            actions=torch.as_tensor(actions, device=device),
            source_id=torch.as_tensor(source, device=device),
            t_idx=torch.as_tensor(t_idx, dtype=torch.long, device=device),
            indices=indices,
        )


class AmbientScalarSampler(ChunkSampler):
    def __init__(
        self,
        chunks: list[MazeChunk],
        schedule: VPSchedule,
        tmin_sigma_scalar: float = 0.074,
        tmin_idx_scalar: int | None = None,
        strict_after_tmin: bool = False,
        seed: int = 0,
        alpha_p: float = 0.019,
    ) -> None:
        super().__init__(chunks, schedule, source_weights={0: alpha_p, 1: 1.0 - alpha_p}, seed=seed)
        self.tmin_idx = int(tmin_idx_scalar) if tmin_idx_scalar is not None else schedule.sigma_to_t_idx(tmin_sigma_scalar)
        self.strict_after_tmin = bool(strict_after_tmin)

    def admissible(self, chunk: MazeChunk, t_idx: int) -> bool:
        if chunk.source_id == 0:
            return True
        if self.strict_after_tmin:
            return t_idx > self.tmin_idx
        return t_idx >= self.tmin_idx


class SRTminSampler(ChunkSampler):
    def __init__(
        self,
        chunks: list[MazeChunk],
        schedule: VPSchedule,
        seed: int = 0,
        alpha_p: float = 0.019,
    ) -> None:
        super().__init__(chunks, schedule, source_weights={0: alpha_p, 1: 1.0 - alpha_p}, seed=seed)

    def admissible(self, chunk: MazeChunk, t_idx: int) -> bool:
        if chunk.source_id == 0:
            return True
        if chunk.sr_tmin_idx is None:
            raise ValueError("SRTminSampler requires chunks to have sr_tmin_idx")
        return t_idx >= chunk.sr_tmin_idx


def source_weights_for_method(method: str, alpha_p: float = 0.019) -> dict[int, float]:
    if method == "gcs_only":
        return {0: 1.0}
    if method in {"rrt_only", "rrt_only_freqmask"}:
        return {1: 1.0}
    if method in {
        "cotrain",
        "ambient_scalar",
        "ambient_sampler_x0_mse",
        "ambient_scalar_ambient_loss",
        "sr_tmin",
        "sr_freqmask",
        "sr_freqmask_visibility_only",
        "sr_freqmask_compatibility_only",
        "sr_freqmask_lowfreq_only",
        "sr_freqmask_constant_lowpass_mask",
        "sr_freqmask_random_mask_same_density",
        "sr_freqmask_shuffled_clean_stats",
        "sr_freqmask_shuffled_target_residuals",
        "sr_full",
    }:
        return {0: alpha_p, 1: 1.0 - alpha_p}
    raise ValueError(f"unknown method: {method}")
