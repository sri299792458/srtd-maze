from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from srtd.diffusion.schedules import VPSchedule
from srtd.spectral.residual import spectral_residual
from srtd.spectral.snr import visibility_mask


@dataclass
class CleanSpectralStats:
    clean_log_amp_median: torch.Tensor
    clean_log_amp_q90: torch.Tensor
    clean_residual_q95: torch.Tensor
    clean_psd_mean: torch.Tensor
    clean_bad_visible_q95: torch.Tensor | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path) -> "CleanSpectralStats":
        data = torch.load(path, map_location="cpu")
        return cls(**data)


def fit_clean_spectral_stats(
    clean_actions: torch.Tensor,
    schedule: VPSchedule | None = None,
    kernel_size: int = 3,
    eps: float = 1e-6,
    clean_bad_visible_quantile: float = 0.95,
    global_band_cutoff_norm: float = 0.2,
    snr_tau: float = 1.0,
    visibility_temperature: float = 0.5,
) -> CleanSpectralStats:
    out = spectral_residual(clean_actions, kernel_size=kernel_size, eps=eps)
    log_amp = out["log_amp"]
    residual = out["residual"]
    psd = out["X"].abs().square()

    stats = CleanSpectralStats(
        clean_log_amp_median=torch.quantile(log_amp, 0.5, dim=0),
        clean_log_amp_q90=torch.quantile(log_amp, 0.90, dim=0),
        clean_residual_q95=torch.quantile(residual, 0.95, dim=0),
        clean_psd_mean=psd.mean(dim=0),
    )
    if schedule is not None:
        bad_visible = bad_visible_scores(
            clean_actions,
            stats,
            schedule,
            kernel_size=kernel_size,
            eps=eps,
            global_band_cutoff_norm=global_band_cutoff_norm,
            snr_tau=snr_tau,
            visibility_temperature=visibility_temperature,
        )
        stats.clean_bad_visible_q95 = torch.quantile(
            bad_visible,
            clean_bad_visible_quantile,
            dim=0,
        )
    return stats


def bad_residual(
    actions: torch.Tensor,
    clean_stats: CleanSpectralStats,
    margin: float = 0.0,
    global_band_cutoff_norm: float = 0.2,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    residual = spectral_residual(actions, kernel_size=kernel_size, eps=eps)["residual"]
    threshold = clean_stats.clean_residual_q95.to(actions.device, actions.dtype)
    bad = (residual - threshold - margin).relu()
    f = bad.shape[1]
    freq_norm = torch.linspace(0, 1, f, device=actions.device, dtype=actions.dtype)
    band = (freq_norm > global_band_cutoff_norm).view(1, f, 1)
    return bad * band


def bad_visible_scores(
    actions: torch.Tensor,
    clean_stats: CleanSpectralStats,
    schedule: VPSchedule,
    margin: float = 0.0,
    global_band_cutoff_norm: float = 0.2,
    kernel_size: int = 3,
    eps: float = 1e-6,
    snr_tau: float = 1.0,
    visibility_temperature: float = 0.5,
) -> torch.Tensor:
    bad = bad_residual(
        actions,
        clean_stats,
        margin=margin,
        global_band_cutoff_norm=global_band_cutoff_norm,
        kernel_size=kernel_size,
        eps=eps,
    )
    visible = visibility_mask(
        clean_stats.clean_psd_mean.to(actions.device, actions.dtype),
        schedule.sigma.to(actions.device, actions.dtype),
        tau=snr_tau,
        temperature=visibility_temperature,
        eps=eps,
    )
    numerator = torch.einsum("bfd,tfd->bt", bad, visible)
    denominator = visible.sum(dim=(1, 2)).clamp_min(eps).view(1, -1)
    return numerator / denominator

