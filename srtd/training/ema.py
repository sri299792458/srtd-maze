from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_p, model_p in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_p.mul_(self.decay).add_(model_p, alpha=1.0 - self.decay)

