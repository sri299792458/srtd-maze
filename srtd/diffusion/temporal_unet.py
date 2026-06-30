from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(t_idx: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=t_idx.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t_idx.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TemporalUNet(nn.Module):
    """Small conditional temporal ConvNet with the same interface as a U-Net policy."""

    def __init__(
        self,
        action_dim: int = 2,
        obs_dim: int = 6,
        base_channels: int = 64,
        time_embed_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        cond_dim = obs_dim + time_embed_dim
        self.time_embed_dim = time_embed_dim
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
        )
        layers: list[nn.Module] = [
            nn.Conv1d(action_dim + base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Conv1d(base_channels, action_dim, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, actions_t: torch.Tensor, obs: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        time_emb = sinusoidal_embedding(t_idx, self.time_embed_dim).to(actions_t.dtype)
        cond = self.cond(torch.cat([obs, time_emb], dim=-1))
        cond_seq = cond[:, :, None].expand(-1, -1, actions_t.shape[1])
        x = torch.cat([actions_t.transpose(1, 2), cond_seq], dim=1)
        return self.net(x).transpose(1, 2)

