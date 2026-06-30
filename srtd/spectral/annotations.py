from __future__ import annotations

import numpy as np
import torch

from srtd.data.maze2d_dataset import MazeChunk
from srtd.diffusion.schedules import VPSchedule
from srtd.spectral.envelope import CleanSpectralStats, bad_visible_scores


def annotate_sr_tmin(
    chunks: list[MazeChunk],
    clean_stats: CleanSpectralStats,
    schedule: VPSchedule,
    eps: float = 1e-6,
    kernel_size: int = 3,
    bad_residual_margin: float = 0.0,
    global_band_cutoff_norm: float = 0.2,
    snr_tau: float = 1.0,
    visibility_temperature: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    actions = torch.as_tensor(np.stack([c.actions for c in chunks]), dtype=torch.float32)
    scores = bad_visible_scores(
        actions,
        clean_stats,
        schedule,
        margin=bad_residual_margin,
        global_band_cutoff_norm=global_band_cutoff_norm,
        kernel_size=kernel_size,
        eps=eps,
        snr_tau=snr_tau,
        visibility_temperature=visibility_temperature,
    )
    clean_threshold = clean_stats.clean_bad_visible_q95
    if clean_threshold is None:
        clean_scores = scores[torch.as_tensor([c.source_id == 0 for c in chunks])]
        clean_threshold = torch.quantile(clean_scores, 0.95, dim=0)

    tmin = np.full(len(chunks), schedule.train_steps, dtype=np.int64)
    scores_np = scores.detach().cpu().numpy()
    threshold_np = clean_threshold.detach().cpu().numpy()
    for i, chunk in enumerate(chunks):
        if chunk.source_id == 0:
            tmin[i] = 0
        else:
            admissible = np.flatnonzero(scores_np[i] <= threshold_np)
            tmin[i] = int(admissible[0]) if len(admissible) else schedule.train_steps
        chunk.sr_tmin_idx = int(tmin[i])
        chunk.sr_bad_score = float(scores_np[i].min())
    return tmin, scores_np

