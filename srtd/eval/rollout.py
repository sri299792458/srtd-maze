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
    finite_difference_squared_acceleration,
    finite_difference_squared_jerk,
    generated_high_freq_residual_energy,
    mean_abs_turn_rate,
    mean_target_jump,
    path_length,
)


@dataclass
class RolloutResult:
    path: np.ndarray
    success: bool
    collision: bool
    collision_padded: bool
    collision_unpadded: bool
    path_length: float
    avg_squared_acceleration: float
    finite_difference_acceleration: float
    squared_jerk: float
    mean_abs_turn_rate: float
    min_clearance_padded: float
    min_clearance_unpadded: float
    action_target_jump: float
    endpoint_error: float
    hf_residual_energy: float
    steps: int


def _result(
    path: list[np.ndarray],
    raw_targets: list[np.ndarray],
    env: MazeEnv,
    goal: np.ndarray,
    success: bool,
    collision_padded: bool,
    collision_unpadded: bool,
    primary_collision_padded: bool,
    dt: float,
) -> RolloutResult:
    arr = np.asarray(path, dtype=np.float32)
    raw = np.asarray(raw_targets, dtype=np.float32) if raw_targets else np.zeros((0, 2), dtype=np.float32)
    collision = collision_padded if primary_collision_padded else collision_unpadded
    return RolloutResult(
        path=arr,
        success=success,
        collision=collision,
        collision_padded=collision_padded,
        collision_unpadded=collision_unpadded,
        path_length=path_length(arr),
        avg_squared_acceleration=average_squared_acceleration(arr, dt=dt),
        finite_difference_acceleration=finite_difference_squared_acceleration(arr, dt=dt),
        squared_jerk=finite_difference_squared_jerk(arr, dt=dt),
        mean_abs_turn_rate=mean_abs_turn_rate(arr, dt=dt),
        min_clearance_padded=env.path_min_clearance(arr, padded=True),
        min_clearance_unpadded=env.path_min_clearance(arr, padded=False),
        action_target_jump=mean_target_jump(raw),
        endpoint_error=endpoint_error(arr, goal),
        hf_residual_energy=generated_high_freq_residual_energy(normalize_xy(arr)),
        steps=len(arr) - 1,
    )


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
    execution_mode: str = "raw",
    lowpass_alpha: float = 0.35,
    interpolation_steps: int = 1,
    primary_collision_padded: bool = True,
) -> RolloutResult:
    model.eval()
    device = torch.device(device)
    max_model_calls = max(1, int(timeout_s / (dt * execute_horizon)))
    prev = start.astype(np.float32)
    curr = start.astype(np.float32)
    path = [curr.copy()]
    raw_targets: list[np.ndarray] = []
    collision_padded = False
    collision_unpadded = False
    filter_state = curr.copy()
    interpolation_steps = max(1, int(interpolation_steps))
    path_dt = dt / interpolation_steps

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
            raw_target = target.astype(np.float32)
            raw_targets.append(raw_target.copy())
            if execution_mode == "filtered":
                filter_state = lowpass_alpha * raw_target + (1.0 - lowpass_alpha) * filter_state
                command = filter_state.astype(np.float32)
            elif execution_mode == "raw":
                command = raw_target
                filter_state = command.copy()
            else:
                raise ValueError("execution_mode must be 'raw' or 'filtered'")

            segment_points = np.linspace(curr, command, interpolation_steps + 1, dtype=np.float32)[1:]
            for point in segment_points:
                padded_free = env.segment_is_free(curr, point, padded=True)
                unpadded_free = env.segment_is_free(curr, point, padded=False)
                collision_padded = collision_padded or not padded_free
                collision_unpadded = collision_unpadded or not unpadded_free
                primary_free = padded_free if primary_collision_padded else unpadded_free
                if not primary_free:
                    path.append(point.astype(np.float32))
                    return _result(
                        path,
                        raw_targets,
                        env,
                        goal,
                        success=False,
                        collision_padded=collision_padded,
                        collision_unpadded=collision_unpadded,
                        primary_collision_padded=primary_collision_padded,
                        dt=path_dt,
                    )
                prev = curr
                curr = point.astype(np.float32)
                path.append(curr.copy())
                if np.linalg.norm(curr - goal) <= success_radius_m:
                    return _result(
                        path,
                        raw_targets,
                        env,
                        goal,
                        success=True,
                        collision_padded=collision_padded,
                        collision_unpadded=collision_unpadded,
                        primary_collision_padded=primary_collision_padded,
                        dt=path_dt,
                    )

    return _result(
        path,
        raw_targets,
        env,
        goal,
        success=False,
        collision_padded=collision_padded,
        collision_unpadded=collision_unpadded,
        primary_collision_padded=primary_collision_padded,
        dt=path_dt,
    )
