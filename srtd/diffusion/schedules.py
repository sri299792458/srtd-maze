from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VPSchedule:
    sigma: torch.Tensor

    @classmethod
    def cosine(cls, train_steps: int = 100, sigma_min: float = 1e-4, sigma_max: float = 0.999) -> "VPSchedule":
        steps = torch.linspace(0.0, 1.0, train_steps)
        sigma = torch.sin(0.5 * torch.pi * steps)
        sigma = sigma_min + (sigma_max - sigma_min) * sigma
        return cls(sigma=sigma.float())

    @property
    def train_steps(self) -> int:
        return int(self.sigma.numel())

    def alpha(self, t_idx: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma.to(t_idx.device)[t_idx]
        return torch.sqrt((1.0 - sigma.square()).clamp(min=1e-8))

    def sigma_at(self, t_idx: torch.Tensor) -> torch.Tensor:
        return self.sigma.to(t_idx.device)[t_idx]

    def sigma_to_t_idx(self, sigma_threshold: float) -> int:
        idx = torch.nonzero(self.sigma >= sigma_threshold)
        if idx.numel() == 0:
            return self.train_steps
        return int(idx[0].item())

