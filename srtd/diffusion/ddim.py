from __future__ import annotations

import torch

from srtd.diffusion.schedules import VPSchedule


def ddim_x0_step(
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_prev: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Deterministic VP-DDIM step for an x0-predicting model."""
    sigma_t = sigma_t.to(device=x_t.device, dtype=x_t.dtype)
    sigma_prev = sigma_prev.to(device=x_t.device, dtype=x_t.dtype)
    alpha_t = torch.sqrt((1.0 - sigma_t.square()).clamp_min(eps))
    alpha_prev = torch.sqrt((1.0 - sigma_prev.square()).clamp_min(eps))
    eps_hat = (x_t - alpha_t * x0_pred) / sigma_t.clamp_min(eps)
    return alpha_prev * x0_pred + sigma_prev * eps_hat


@torch.no_grad()
def ddim_sample(
    model,
    obs: torch.Tensor,
    schedule: VPSchedule,
    horizon: int = 16,
    action_dim: int = 2,
    inference_steps: int = 10,
) -> torch.Tensor:
    device = obs.device
    x = torch.randn(obs.shape[0], horizon, action_dim, device=device, dtype=obs.dtype)
    t_indices = torch.linspace(schedule.train_steps - 1, 0, inference_steps, device=device).long()
    sigmas = schedule.sigma.to(device=device, dtype=obs.dtype)
    for idx, t in enumerate(t_indices):
        t_batch = torch.full((obs.shape[0],), int(t.item()), device=device, dtype=torch.long)
        x0 = model(x, obs, t_batch)
        if idx == len(t_indices) - 1:
            x = x0
            break
        sigma_t = sigmas[int(t.item())]
        sigma_prev = sigmas[int(t_indices[idx + 1].item())]
        x = ddim_x0_step(x, x0, sigma_t, sigma_prev)
    return x
