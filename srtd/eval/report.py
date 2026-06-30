from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from srtd.diffusion.schedules import VPSchedule
from srtd.diffusion.temporal_unet import TemporalUNet
from srtd.eval.maze_env import MazeEnv
from srtd.eval.rollout import rollout_policy


def _binomial_ci(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    se = np.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def _bootstrap_ci(values: list[float], rng: np.random.Generator, reps: int = 1000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(reps)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _load_model(run_dir: Path, device: torch.device):
    ckpt = torch.load(run_dir / "checkpoint_last.pt", map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model = TemporalUNet(
        action_dim=int(model_cfg.get("action_dim", 2)),
        obs_dim=int(model_cfg.get("obs_dim", 6)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    return model, cfg


def evaluate_run(run_dir: Path, starts: np.ndarray, goals: np.ndarray, device: torch.device) -> dict:
    model, cfg = _load_model(run_dir, device)
    env = MazeEnv.from_yaml(cfg["dataset"].get("maze_yaml", "assets/maze2d_default.yaml"))
    schedule = VPSchedule.cosine(train_steps=int(cfg["diffusion"]["train_steps"]))
    schedule.sigma = schedule.sigma.to(device)
    eval_cfg = cfg["eval"]
    dataset_cfg = cfg["dataset"]
    diffusion_cfg = cfg["diffusion"]
    results = [
        rollout_policy(
            model,
            env,
            schedule,
            start,
            goal,
            policy_horizon=int(dataset_cfg["policy_horizon"]),
            execute_horizon=int(dataset_cfg["execute_horizon"]),
            inference_steps=int(diffusion_cfg.get("inference_steps", 10)),
            success_radius_m=float(eval_cfg.get("success_radius_m", 0.15)),
            timeout_s=float(eval_cfg.get("timeout_s", 30.0)),
            dt=float(eval_cfg.get("dt", 0.1)),
            device=device,
        )
        for start, goal in zip(starts, goals, strict=True)
    ]
    successes = [r for r in results if r.success]
    success_rate = len(successes) / len(results)
    ci_low, ci_high = _binomial_ci(len(successes), len(results))
    rng = np.random.default_rng(0)
    smooth_values = [r.avg_squared_acceleration for r in successes]
    smooth_low, smooth_high = _bootstrap_ci(smooth_values, rng)
    return {
        "policy": cfg["method"],
        "run_dir": str(run_dir),
        "seed": cfg.get("seed", 0),
        "success_rate": success_rate,
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "smoothness_mean": float(np.mean(smooth_values)) if smooth_values else float("nan"),
        "smoothness_bootstrap_ci_low": smooth_low,
        "smoothness_bootstrap_ci_high": smooth_high,
        "path_length_mean": float(np.mean([r.path_length for r in successes])) if successes else float("nan"),
        "collision_rate": float(np.mean([r.collision for r in results])),
        "endpoint_error": float(np.mean([r.endpoint_error for r in results])),
        "hf_residual_energy_mean": float(np.mean([r.hf_residual_energy for r in results])),
        "mean_num_model_steps_to_success": float(np.mean([r.steps for r in successes])) if successes else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument("--out", default="runs/maze2d_report")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dirs = [Path(p) for pattern in args.runs for p in sorted(Path().glob(pattern))]
    if not run_dirs:
        raise FileNotFoundError("no run directories matched")

    with (run_dirs[0] / "config.json").open("r", encoding="utf-8") as f:
        first_cfg = json.load(f)
    env = MazeEnv.from_yaml(first_cfg["dataset"].get("maze_yaml", "assets/maze2d_default.yaml"))
    rng = np.random.default_rng(args.seed)
    starts, goals = [], []
    while len(starts) < args.num_trials:
        start = env.sample_free(rng)
        goal = env.sample_free(rng)
        if np.linalg.norm(goal - start) >= 2.0:
            starts.append(start)
            goals.append(goal)
    starts_arr = np.asarray(starts, dtype=np.float32)
    goals_arr = np.asarray(goals, dtype=np.float32)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "shared_trials.npz", starts=starts_arr, goals=goals_arr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [evaluate_run(run_dir, starts_arr, goals_arr, device) for run_dir in run_dirs]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()

