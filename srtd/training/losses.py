from __future__ import annotations

import torch
import torch.nn.functional as F

from srtd.diffusion.schedules import VPSchedule
from srtd.spectral.envelope import CleanSpectralStats
from srtd.spectral.fft_ops import rfft_to_full_fft_weights
from srtd.spectral.residual import spectral_residual
from srtd.spectral.snr import visibility_mask


def weighted_time_mse(pred: torch.Tensor, target: torch.Tensor, time_weight: torch.Tensor) -> torch.Tensor:
    err = (pred - target).square().sum(dim=-1)
    return (err * time_weight).sum() / (time_weight.sum() * target.shape[-1]).clamp_min(1e-8)


def spectral_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    error = torch.fft.fft(pred - target, dim=1, norm="ortho")
    loss_fd = (error.real.square() + error.imag.square()) * weights
    return loss_fd.sum() / weights.sum().clamp_min(1e-8)


def combine_source_losses(
    p_loss: torch.Tensor | None,
    q_loss: torch.Tensor | None,
    n_p: int,
    n_q: int,
) -> torch.Tensor:
    losses = []
    if p_loss is not None:
        losses.append((p_loss, n_p))
    if q_loss is not None:
        losses.append((q_loss, n_q))
    if not losses:
        raise ValueError("at least one source loss is required")
    total = sum(count for _, count in losses)
    return sum(loss * count for loss, count in losses) / max(total, 1)


def q_frequency_weights(
    target: torch.Tensor,
    t_idx: torch.Tensor,
    clean_stats: CleanSpectralStats,
    schedule: VPSchedule,
    compat_temperature: float = 0.25,
    low_freq_floor: float = 0.25,
    global_band_cutoff_norm: float = 0.2,
    bad_residual_margin: float = 0.0,
    snr_tau: float = 1.0,
    visibility_temperature: float = 0.5,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    residual = spectral_residual(target, kernel_size=kernel_size, eps=eps)["residual"]
    threshold = clean_stats.clean_residual_q95.to(target.device, target.dtype)
    signed_margin = threshold + bad_residual_margin - residual
    compatibility = torch.sigmoid(signed_margin / compat_temperature)
    b, f, _ = compatibility.shape
    freq_norm = torch.linspace(0, 1, f, device=target.device, dtype=target.dtype)
    low = freq_norm <= global_band_cutoff_norm
    compatibility[:, low, :] = 1.0

    sigma = schedule.sigma.to(target.device, target.dtype)[t_idx]
    visible = visibility_mask(
        clean_stats.clean_psd_mean.to(target.device, target.dtype),
        sigma,
        tau=snr_tau,
        temperature=visibility_temperature,
        eps=eps,
    )
    weights = (visible * compatibility).clamp(0.0, 1.0)
    weights[:, low, :] = weights[:, low, :].clamp_min(low_freq_floor)
    return weights


def sr_freqmask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_id: torch.Tensor,
    t_idx: torch.Tensor,
    clean_stats: CleanSpectralStats,
    schedule: VPSchedule,
    **kwargs,
) -> torch.Tensor:
    p_loss = None
    q_loss = None
    p_mask = source_id == 0
    q_mask = source_id == 1
    if p_mask.any():
        p_loss = F.mse_loss(pred[p_mask], target[p_mask])
    if q_mask.any():
        rfft_w = q_frequency_weights(
            target[q_mask],
            t_idx[q_mask],
            clean_stats,
            schedule,
            **kwargs,
        )
        full_w = rfft_to_full_fft_weights(rfft_w, horizon=target.shape[1])
        q_loss = spectral_mse(pred[q_mask], target[q_mask], full_w)
    return combine_source_losses(
        p_loss,
        q_loss,
        int(p_mask.sum().item()),
        int(q_mask.sum().item()),
    )


def sr_full_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_id: torch.Tensor,
    t_idx: torch.Tensor,
    clean_stats: CleanSpectralStats,
    schedule: VPSchedule,
    beta_clean_saliency: float = 0.25,
    saliency_clip: float = 5.0,
    lambda_p_freq: float = 0.1,
    q_loss_weight: float = 1.0,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    p_loss = None
    q_loss = None
    p_mask = source_id == 0
    q_mask = source_id == 1
    if p_mask.any():
        saliency = spectral_residual(target[p_mask], eps=eps)["saliency_scalar"]
        saliency = saliency / (saliency.mean(dim=1, keepdim=True) + eps)
        time_weight = 1.0 + beta_clean_saliency * saliency.clamp(max=saliency_clip)
        loss_p_time = weighted_time_mse(pred[p_mask], target[p_mask], time_weight)
        all_w = torch.ones_like(pred[p_mask])
        loss_p_freq = spectral_mse(pred[p_mask], target[p_mask], all_w)
        p_loss = loss_p_time + lambda_p_freq * loss_p_freq
    if q_mask.any():
        rfft_w = q_frequency_weights(
            target[q_mask],
            t_idx[q_mask],
            clean_stats,
            schedule,
            eps=eps,
            **kwargs,
        )
        full_w = rfft_to_full_fft_weights(rfft_w, horizon=target.shape[1])
        q_loss = q_loss_weight * spectral_mse(pred[q_mask], target[q_mask], full_w)
    return combine_source_losses(
        p_loss,
        q_loss,
        int(p_mask.sum().item()),
        int(q_mask.sum().item()),
    )


def add_vp_noise(
    target: torch.Tensor,
    t_idx: torch.Tensor,
    schedule: VPSchedule,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise is None:
        noise = torch.randn_like(target)
    sigma = schedule.sigma_at(t_idx).to(target.device, target.dtype)
    alpha = torch.sqrt((1.0 - sigma.square()).clamp(min=1e-8))
    noised = alpha[:, None, None] * target + sigma[:, None, None] * noise
    return noised, noise
