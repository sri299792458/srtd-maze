from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import beta

from srtd.diffusion.schedules import make_vp_schedule
from srtd.diffusion.temporal_unet import TemporalUNet
from srtd.eval.bundle import create_repro_bundle
from srtd.eval.maze_env import MazeEnv
from srtd.eval.rollout import rollout_policy

POLICY_ORDER = {
    "gcs_only": 0,
    "rrt_only": 1,
    "cotrain": 2,
    "ambient_scalar": 3,
    "ambient_sampler_x0_mse": 3,
    "ambient_scalar_ambient_loss": 4,
    "sr_tmin": 5,
    "sr_freqmask": 6,
    "sr_freqmask_visibility_only": 7,
    "sr_freqmask_compatibility_only": 8,
    "sr_freqmask_lowfreq_only": 9,
    "sr_freqmask_shuffled_clean_stats": 10,
    "rrt_only_freqmask": 11,
    "sr_full": 12,
}


def _binomial_ci(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    alpha = 0.05
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    high = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return low, high


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


def evaluate_run(
    run_dir: Path,
    starts: np.ndarray,
    goals: np.ndarray,
    device: torch.device,
    save_rollouts: int = 0,
    execution_mode: str = "raw",
    lowpass_alpha: float = 0.35,
    interpolation_steps: int = 1,
    primary_collision_padded: bool = True,
) -> tuple[dict, list[np.ndarray]]:
    model, cfg = _load_model(run_dir, device)
    env = MazeEnv.from_yaml(cfg["dataset"].get("maze_yaml", "assets/maze2d_default.yaml"))
    schedule = make_vp_schedule(cfg["diffusion"])
    schedule.sigma = schedule.sigma.to(device)
    eval_cfg = cfg["eval"]
    dataset_cfg = cfg["dataset"]
    diffusion_cfg = cfg["diffusion"]
    results = []
    for start, goal in zip(starts, goals, strict=True):
        results.append(
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
            execution_mode=execution_mode,
            lowpass_alpha=lowpass_alpha,
            interpolation_steps=interpolation_steps,
            primary_collision_padded=primary_collision_padded,
        )
        )
    successes = [r for r in results if r.success]
    success_rate = len(successes) / len(results)
    ci_low, ci_high = _binomial_ci(len(successes), len(results))
    rng = np.random.default_rng(0)
    smooth_values = [r.avg_squared_acceleration for r in successes]
    smooth_low, smooth_high = _bootstrap_ci(smooth_values, rng)
    row = {
        "policy": cfg["method"],
        "run_dir": str(run_dir),
        "seed": cfg.get("seed", 0),
        "execution_mode": execution_mode,
        "lowpass_alpha": lowpass_alpha,
        "interpolation_steps": interpolation_steps,
        "primary_collision_padding": "padded" if primary_collision_padded else "unpadded",
        "success_rate": success_rate,
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "smoothness_mean": float(np.mean(smooth_values)) if smooth_values else float("nan"),
        "smoothness_bootstrap_ci_low": smooth_low,
        "smoothness_bootstrap_ci_high": smooth_high,
        "finite_difference_acceleration_mean": float(np.mean([r.finite_difference_acceleration for r in successes])) if successes else float("nan"),
        "squared_jerk_mean": float(np.mean([r.squared_jerk for r in successes])) if successes else float("nan"),
        "mean_abs_turn_rate": float(np.mean([r.mean_abs_turn_rate for r in successes])) if successes else float("nan"),
        "min_clearance_padded_mean": float(np.mean([r.min_clearance_padded for r in results])),
        "min_clearance_unpadded_mean": float(np.mean([r.min_clearance_unpadded for r in results])),
        "action_target_jump_mean": float(np.mean([r.action_target_jump for r in results])),
        "path_length_mean": float(np.mean([r.path_length for r in successes])) if successes else float("nan"),
        "collision_rate": float(np.mean([r.collision for r in results])),
        "collision_rate_padded": float(np.mean([r.collision_padded for r in results])),
        "collision_rate_unpadded": float(np.mean([r.collision_unpadded for r in results])),
        "endpoint_error": float(np.mean([r.endpoint_error for r in results])),
        "hf_residual_energy_mean": float(np.mean([r.hf_residual_energy for r in results])),
        "mean_num_model_steps_to_success": float(np.mean([r.steps for r in successes])) if successes else float("nan"),
    }
    paths = [r.path for r in results[:save_rollouts]]
    return row, paths


def _policy_sort_key(row: dict) -> tuple[int, str]:
    policy = str(row["policy"])
    return POLICY_ORDER.get(policy, 999), policy


def plot_summary_figures(rows: list[dict], out: Path) -> None:
    ordered = sorted(rows, key=_policy_sort_key)
    policies = [str(r["policy"]) for r in ordered]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for row in ordered:
        smooth = float(row["smoothness_mean"])
        success = float(row["success_rate"])
        if np.isfinite(smooth):
            ax.scatter(smooth, success, s=70)
            ax.text(smooth, success, str(row["policy"]), fontsize=8)
    ax.set_xlabel("average squared acceleration, success only")
    ax.set_ylabel("success rate")
    ax.set_title("Success vs smoothness")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "success_vs_smoothness_pareto.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    values = [float(r["hf_residual_energy_mean"]) for r in ordered]
    ax.bar(policies, values)
    ax.set_ylabel("mean high-frequency residual energy")
    ax.set_title("Generated residual energy")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out / "generated_residual_energy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    success = [float(r["success_rate"]) for r in ordered]
    collision = [float(r["collision_rate"]) for r in ordered]
    x = np.arange(len(policies))
    width = 0.38
    ax.bar(x - width / 2, success, width, label="success")
    ax.bar(x + width / 2, collision, width, label="collision")
    ax.set_xticks(x, policies, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Success and collision rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "success_collision_rates.png", dpi=160)
    plt.close(fig)


def plot_rollout_grid(
    paths_by_policy: dict[str, list[np.ndarray]],
    starts: np.ndarray,
    goals: np.ndarray,
    env: MazeEnv,
    out: Path,
    max_trials: int = 5,
) -> None:
    policies = sorted(paths_by_policy, key=lambda p: POLICY_ORDER.get(p, 999))
    if not policies:
        return
    cols = min(max_trials, min(len(v) for v in paths_by_policy.values()))
    if cols == 0:
        return

    fig, axes = plt.subplots(
        len(policies),
        cols,
        figsize=(2.4 * cols, 2.2 * len(policies)),
        squeeze=False,
    )
    xmin, ymin, xmax, ymax = env.bounds
    for r, policy in enumerate(policies):
        for c in range(cols):
            ax = axes[r, c]
            for obs in env.obstacles:
                rect = plt.Rectangle(
                    (obs.xmin, obs.ymin),
                    obs.xmax - obs.xmin,
                    obs.ymax - obs.ymin,
                    color="0.75",
                )
                ax.add_patch(rect)
            path = paths_by_policy[policy][c]
            ax.plot(path[:, 0], path[:, 1], color="#2c7fb8", linewidth=1.5)
            ax.scatter(starts[c, 0], starts[c, 1], c="#31a354", s=20, marker="o")
            ax.scatter(goals[c, 0], goals[c, 1], c="#de2d26", s=24, marker="x")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(policy, fontsize=8)
            if r == 0:
                ax.set_title(f"trial {c}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rollout_grid_same_start_goal.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument("--out", default="runs/maze2d_report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-rollouts", type=int, default=5)
    parser.add_argument("--execution-mode", choices=["raw", "filtered"], default="raw")
    parser.add_argument("--lowpass-alpha", type=float, default=0.35)
    parser.add_argument("--interpolation-steps", type=int, default=1)
    parser.add_argument("--primary-collision-padding", choices=["padded", "unpadded"], default="padded")
    parser.add_argument("--bundle-out", default=None)
    parser.add_argument("--dataset", default=None)
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
    rows: list[dict] = []
    paths_by_policy: dict[str, list[np.ndarray]] = {}
    for run_dir in run_dirs:
        row, paths = evaluate_run(
            run_dir,
            starts_arr,
            goals_arr,
            device,
            save_rollouts=args.save_rollouts,
            execution_mode=args.execution_mode,
            lowpass_alpha=args.lowpass_alpha,
            interpolation_steps=args.interpolation_steps,
            primary_collision_padded=args.primary_collision_padding == "padded",
        )
        rows.append(row)
        paths_by_policy[str(row["policy"])] = paths

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out / "rollout_paths.npz",
        **{
            f"{policy}_{i}": path
            for policy, paths in paths_by_policy.items()
            for i, path in enumerate(paths)
        },
    )
    plot_summary_figures(rows, out)
    plot_rollout_grid(paths_by_policy, starts_arr, goals_arr, env, out)
    with (out / "eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "execution_mode": args.execution_mode,
                "lowpass_alpha": args.lowpass_alpha,
                "interpolation_steps": args.interpolation_steps,
                "primary_collision_padding": args.primary_collision_padding,
                "num_trials": args.num_trials,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    if args.bundle_out:
        create_repro_bundle(
            Path(args.bundle_out),
            Path(args.dataset) if args.dataset else None,
            run_dirs,
            out,
        )
    print(f"wrote {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
