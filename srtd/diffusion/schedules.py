from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VPSchedule:
    sigma: torch.Tensor
    name: str = "custom"

    @classmethod
    def sine_sigma(cls, train_steps: int = 100, sigma_min: float = 1e-4, sigma_max: float = 0.999) -> "VPSchedule":
        steps = torch.linspace(0.0, 1.0, train_steps)
        sigma = torch.sin(0.5 * torch.pi * steps)
        sigma = sigma_min + (sigma_max - sigma_min) * sigma
        return cls(sigma=sigma.float(), name="sine_sigma")

    @classmethod
    def cosine(cls, train_steps: int = 100, sigma_min: float = 1e-4, sigma_max: float = 0.999) -> "VPSchedule":
        return cls.sine_sigma(train_steps=train_steps, sigma_min=sigma_min, sigma_max=sigma_max)

    @classmethod
    def diffusion_policy_cosine(cls, train_steps: int = 100, max_beta: float = 0.999) -> "VPSchedule":
        """VP schedule matching Diffusion Policy's diffusers `squaredcos_cap_v2`.

        Diffusion Policy configures diffusers' DDPMScheduler with
        `beta_schedule: squaredcos_cap_v2`. For VP noise
        `x_t = sqrt(alpha_cumprod_t) x0 + sqrt(1 - alpha_cumprod_t) eps`,
        sigma is `sqrt(1 - alpha_cumprod_t)`.
        """
        betas = []
        for i in range(train_steps):
            t1 = i / train_steps
            t2 = (i + 1) / train_steps
            beta = min(1.0 - _diffusers_alpha_bar(t2) / _diffusers_alpha_bar(t1), max_beta)
            betas.append(beta)
        betas_t = torch.as_tensor(betas, dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1.0 - betas_t, dim=0)
        sigma = torch.sqrt((1.0 - alphas_cumprod).clamp(0.0, 1.0))
        return cls(sigma=sigma.float(), name="diffusion_policy_cosine")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VPSchedule":
        name = str(cfg.get("schedule", "sine_sigma"))
        train_steps = int(cfg.get("train_steps", 100))
        if name in {"cosine", "sine_sigma"}:
            return cls.sine_sigma(
                train_steps=train_steps,
                sigma_min=float(cfg.get("sigma_min", 1e-4)),
                sigma_max=float(cfg.get("sigma_max", 0.999)),
            )
        if name in {"diffusion_policy_cosine", "squaredcos_cap_v2"}:
            return cls.diffusion_policy_cosine(
                train_steps=train_steps,
                max_beta=float(cfg.get("max_beta", 0.999)),
            )
        raise ValueError(f"unknown VP schedule: {name}")

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

    def t_idx_to_sigma(self, t_idx: int) -> float:
        if t_idx < 0 or t_idx >= self.train_steps:
            raise IndexError(f"t_idx {t_idx} outside [0, {self.train_steps})")
        return float(self.sigma[t_idx].item())


def _diffusers_alpha_bar(time_step: float) -> float:
    return torch.cos(torch.as_tensor((time_step + 0.008) / 1.008) * torch.pi / 2).item() ** 2


def make_vp_schedule(cfg: dict[str, Any]) -> VPSchedule:
    return VPSchedule.from_config(cfg)
