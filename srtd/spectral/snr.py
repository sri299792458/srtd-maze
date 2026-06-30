from __future__ import annotations

import torch


def visibility_mask(
    clean_psd_mean: torch.Tensor,
    sigma: torch.Tensor | float,
    tau: float = 1.0,
    temperature: float = 0.5,
    eps: float = 1e-6,
    min_visible_weight: float = 0.0,
) -> torch.Tensor:
    """Soft SNR visibility, returned as [B, F, D]."""
    if not torch.is_tensor(sigma):
        sigma = torch.as_tensor([sigma], dtype=clean_psd_mean.dtype, device=clean_psd_mean.device)
    sigma = sigma.to(device=clean_psd_mean.device, dtype=clean_psd_mean.dtype).reshape(-1)
    sigma2 = sigma.square().clamp(max=1.0 - eps).view(-1, 1, 1)
    signal_scale = (1.0 - sigma2).clamp(min=eps)
    psd = clean_psd_mean.to(dtype=sigma.dtype, device=sigma.device).unsqueeze(0)
    snr = signal_scale * psd / (sigma2 + eps)
    logits = (torch.log(snr + eps) - torch.log(torch.as_tensor(tau, device=sigma.device))) / temperature
    visible = torch.sigmoid(logits)
    if min_visible_weight > 0:
        visible = visible.clamp_min(min_visible_weight)
    return visible


def max_visible_frequency(visible: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Return the largest visible frequency index per batch item."""
    if visible.ndim != 3:
        raise ValueError("visible must have shape [B, F, D]")
    per_freq = visible.max(dim=-1).values >= threshold
    idx = torch.arange(visible.shape[1], device=visible.device)
    return (per_freq * idx.view(1, -1)).max(dim=1).values

