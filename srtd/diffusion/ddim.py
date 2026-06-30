from __future__ import annotations

import torch

from srtd.diffusion.schedules import VPSchedule


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
    for t in t_indices:
        t_batch = torch.full((obs.shape[0],), int(t.item()), device=device, dtype=torch.long)
        x0 = model(x, obs, t_batch)
        if t == 0:
            x = x0
        else:
            prev_t = max(int(t.item()) - max(1, schedule.train_steps // inference_steps), 0)
            sigma = schedule.sigma[prev_t].to(device=device, dtype=obs.dtype)
            alpha = torch.sqrt((1.0 - sigma.square()).clamp(min=1e-8))
            x = alpha * x0
    return x

