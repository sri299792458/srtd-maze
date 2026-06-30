from __future__ import annotations

import torch

from srtd.spectral.fft_ops import avg_pool_freq, rfft_actions


def spectral_residual(
    actions: torch.Tensor,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Hou-Zhang-style residual saliency for trajectory chunks.

    Args:
        actions: Tensor with shape [B, H, D].
    """
    x = rfft_actions(actions)
    amp = torch.abs(x)
    phase = torch.angle(x)
    log_amp = torch.log(amp + eps)
    smooth_log_amp = avg_pool_freq(log_amp, kernel_size=kernel_size)
    residual = log_amp - smooth_log_amp

    residual_complex = torch.exp(residual) * torch.exp(1j * phase)
    saliency_time = torch.fft.irfft(
        residual_complex,
        n=actions.shape[1],
        dim=1,
        norm="ortho",
    ).abs().square()
    saliency_scalar = saliency_time.sum(dim=-1)
    saliency_scalar = saliency_scalar / (
        saliency_scalar.mean(dim=1, keepdim=True) + eps
    )
    return {
        "X": x,
        "log_amp": log_amp,
        "smooth_log_amp": smooth_log_amp,
        "residual": residual,
        "phase": phase,
        "saliency_time": saliency_time,
        "saliency_scalar": saliency_scalar,
    }


def high_frequency_residual_energy(
    actions: torch.Tensor,
    cutoff_norm: float = 0.2,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    out = spectral_residual(actions, kernel_size=kernel_size, eps=eps)
    residual = out["residual"].relu()
    f = residual.shape[1]
    freq_norm = torch.linspace(0, 1, f, device=actions.device, dtype=actions.dtype)
    mask = (freq_norm > cutoff_norm).view(1, f, 1)
    return (residual * mask).mean(dim=(1, 2))

