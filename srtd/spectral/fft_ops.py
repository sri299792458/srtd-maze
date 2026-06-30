from __future__ import annotations

import torch
import torch.nn.functional as F


def rfft_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError("actions must have shape [B, H, D]")
    return torch.fft.rfft(actions, dim=1, norm="ortho")


def irfft_actions(coeffs: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.fft.irfft(coeffs, n=horizon, dim=1, norm="ortho")


def avg_pool_freq(values: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("values must have shape [B, F, D]")
    if kernel_size <= 1:
        return values
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    b, f, d = values.shape
    pad = kernel_size // 2
    x = values.permute(0, 2, 1).reshape(b * d, 1, f)
    mode = "reflect" if f > pad else "replicate"
    x = F.pad(x, (pad, pad), mode=mode)
    pooled = F.avg_pool1d(x, kernel_size=kernel_size, stride=1)
    return pooled.reshape(b, d, f).permute(0, 2, 1)


def rfft_to_full_fft_weights(rfft_weights: torch.Tensor, horizon: int) -> torch.Tensor:
    if rfft_weights.ndim != 3:
        raise ValueError("rfft_weights must have shape [B, F, D]")
    b, f, d = rfft_weights.shape
    expected_f = horizon // 2 + 1
    if f != expected_f:
        raise ValueError(f"expected {expected_f} rFFT bins for horizon {horizon}, got {f}")

    full = rfft_weights.new_empty((b, horizon, d))
    full[:, :f, :] = rfft_weights
    if horizon % 2 == 0:
        tail = rfft_weights[:, 1:-1, :].flip(1)
    else:
        tail = rfft_weights[:, 1:, :].flip(1)
    full[:, f:, :] = tail
    return full

