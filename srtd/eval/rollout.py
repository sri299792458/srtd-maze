from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from srtd.data.normalization import denormalize_xy, normalize_xy
from srtd.diffusion.ddim import ddim_sample
from srtd.diffusion.schedules import VPSchedule
from srtd.eval.maze_env import MazeEnv
from srtd.eval.metrics import (
    average_squared_acceleration,
    endpoint_error,
    generated_high_freq_residual_energy,
    path_length,
)


@dataclass
class RolloutResult:
    path: np.ndarray
    success: bool
    collision: bool
    path_length: float
    avg_squared_acceleration: float
    endpoint_error: float
    hf_residual_energy: float
    steps: int


def rollout_policy(
    model: torch.nn.Module,
    env: MazeEnv,
    schedule: VPSchedule,
    start: np.ndarray,
    goal: np.ndarray,
    policy_horizon: int = 16,
    execute_horizon: int = 8,
    inference_steps: int = 10,
    success_radius_m: float = 0.15,
    timeout_s: float = 30.0,
    dt: float = 0.1,
    device: str | torch.device = "cpu",
) -> RolloutResult:
    model.eval()
    device = torch.device(device)
    max_model_calls = max(1, int(timeout_s / (dt * execute_horizon)))
    prev = start.astype(np.float32)
    curr = start.astype(np.float32)
    path = [curr.copy()]
    collision = False

    for _ in range(max_model_calls):
        obs_np = np.concatenate(
            [
                normalize_xy(prev),
                normalize_xy(curr),
                normalize_xy(goal),
            ]
        ).astype(np.float32)
        obs = torch.as_tensor(obs_np[None], device=device)
        actions_norm = ddim_sample(
            model,
            obs,
            schedule,
            horizon=policy_horizon,
            action_dim=2,
            inference_steps=inference_steps,
        )[0].detach().cpu().numpy()
        actions_m = denormalize_xy(actions_norm)
        for target in actions_m[:execute_horizon]:
            if not env.segment_is_free(curr, target, padded=False):
                collision = True
                path.append(target.astype(np.float32))
                break
            prev = curr
            curr = target.astype(np.float32)
            path.append(curr.copy())
            if np.linalg.norm(curr - goal) <= success_radius_m:
                arr = np.asarray(path, dtype=np.float32)
                return RolloutResult(
                    path=arr,
                    success=True,
                    collision=False,
                    path_length=path_length(arr),
                    avg_squared_acceleration=average_squared_acceleration(arr, dt=dt),
                    endpoint_error=endpoint_error(arr, goal),
                    hf_residual_energy=generated_high_freq_residual_energy(normalize_xy(arr)),
                    steps=len(arr) - 1,
                )
        if collision:
            break

    arr = np.asarray(path, dtype=np.float32)
    return RolloutResult(
        path=arr,
        success=False,
        collision=collision,
        path_length=path_length(arr),
        avg_squared_acceleration=average_squared_acceleration(arr, dt=dt),
        endpoint_error=endpoint_error(arr, goal),
        hf_residual_energy=generated_high_freq_residual_energy(normalize_xy(arr)),
        steps=len(arr) - 1,
    )

