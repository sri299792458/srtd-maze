from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from srtd.config import load_config
from srtd.data.maze2d_dataset import MazeChunk, build_chunks, load_episodes, save_episodes
from srtd.data.maze2d_generate import generate_fallback_episodes
from srtd.diffusion.schedule_report import schedule_report
from srtd.diffusion.schedules import VPSchedule, make_vp_schedule
from srtd.diffusion.temporal_unet import TemporalUNet
from srtd.eval.diagnostics import selected_sr_tmin_table, sr_tmin_usable_fraction_from_tmins
from srtd.eval.maze_env import MazeEnv
from srtd.spectral.annotations import annotate_sr_tmin
from srtd.spectral.envelope import fit_clean_spectral_stats
from srtd.spectral.plotting import (
    plot_psd_gcs_vs_rrt,
    plot_residual_saliency_examples,
    plot_sr_tmin_hist,
    plot_visibility,
)
from srtd.spectral.snr import visibility_mask
from srtd.training.ema import EMA
from srtd.training.losses import add_vp_ambient_noise, add_vp_noise, sr_freqmask_loss, sr_full_loss, vp_ambient_x0_loss
from srtd.training.samplers import AmbientScalarSampler, ChunkSampler, SRTminSampler, source_weights_for_method

SR_FREQMASK_METHODS = {
    "sr_freqmask",
    "sr_freqmask_visibility_only",
    "sr_freqmask_compatibility_only",
    "sr_freqmask_lowfreq_only",
    "sr_freqmask_constant_lowpass_mask",
    "sr_freqmask_random_mask_same_density",
    "sr_freqmask_shuffled_clean_stats",
    "sr_freqmask_shuffled_target_residuals",
    "rrt_only_freqmask",
}

SR_FREQMASK_METHOD_TO_MASK_MODE = {
    "sr_freqmask": "full",
    "rrt_only_freqmask": "full",
    "sr_freqmask_visibility_only": "visibility_only",
    "sr_freqmask_compatibility_only": "compatibility_only",
    "sr_freqmask_lowfreq_only": "lowfreq_only",
    "sr_freqmask_constant_lowpass_mask": "constant_lowpass_mask",
    "sr_freqmask_random_mask_same_density": "random_mask_same_density",
    "sr_freqmask_shuffled_clean_stats": "shuffled_clean_stats",
    "sr_freqmask_shuffled_target_residuals": "shuffled_target_residuals",
}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_id(method: str, seed: int, run_label: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = run_label or method
    return f"maze2d_{label}_seed{seed}_{stamp}"


def _load_or_generate(cfg: dict, seed: int):
    dataset_cfg = cfg["dataset"]
    path = dataset_cfg.get("path")
    if path and Path(path).exists():
        print(f"loading dataset from {path}", flush=True)
        return load_episodes(path)
    cache_path = dataset_cfg.get("cache_path")
    if cache_path is None:
        cache_path = (
            Path("data/generated")
            / f"{dataset_cfg.get('name', 'maze2d')}_seed{seed}_"
            f"p{dataset_cfg.get('num_clean_episodes', 50)}_"
            f"q{dataset_cfg.get('num_rrt_episodes', 5000)}_"
            f"h{dataset_cfg.get('episode_horizon', 100)}.npz"
        )
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"loading cached fallback dataset from {cache_path}", flush=True)
        return load_episodes(cache_path)
    if not dataset_cfg.get("fallback_generate", True):
        raise FileNotFoundError(f"dataset path not found: {path}")
    env = MazeEnv.from_yaml(dataset_cfg.get("maze_yaml", "assets/maze2d_default.yaml"))
    print(f"generating fallback dataset at {cache_path}", flush=True)
    episodes = generate_fallback_episodes(
        env,
        num_clean=int(dataset_cfg.get("num_clean_episodes", 50)),
        num_rrt=int(dataset_cfg.get("num_rrt_episodes", 5000)),
        horizon=int(dataset_cfg.get("episode_horizon", 100)),
        seed=seed,
        verbose=True,
    )
    save_episodes(cache_path, episodes)
    print(f"cached fallback dataset at {cache_path}", flush=True)
    return episodes


def _make_sampler(method: str, chunks: list[MazeChunk], schedule: VPSchedule, cfg: dict, seed: int):
    alpha_p = float(cfg["training"].get("alpha_p", 0.019))
    if method in {"ambient_scalar", "ambient_sampler_x0_mse", "ambient_scalar_ambient_loss"}:
        return AmbientScalarSampler(
            chunks,
            schedule,
            tmin_sigma_scalar=float(cfg["training"].get("tmin_sigma_scalar", 0.074)),
            tmin_idx_scalar=cfg["training"].get("tmin_idx_scalar"),
            strict_after_tmin=method == "ambient_scalar_ambient_loss",
            seed=seed,
            alpha_p=alpha_p,
        )
    if method == "sr_tmin":
        return SRTminSampler(chunks, schedule, seed=seed, alpha_p=alpha_p)
    return ChunkSampler(
        chunks,
        schedule,
        source_weights=source_weights_for_method(method, alpha_p=alpha_p),
        seed=seed,
    )


def prepare_chunks_and_stats(cfg: dict, run_dir: Path, seed: int):
    episodes = _load_or_generate(cfg, seed)
    dataset_cfg = cfg["dataset"]
    chunks = build_chunks(
        episodes,
        policy_horizon=int(dataset_cfg["policy_horizon"]),
        obs_horizon=int(dataset_cfg.get("obs_horizon", 2)),
        stride=int(dataset_cfg.get("stride", 4)),
    )
    actions = torch.as_tensor(np.stack([c.actions for c in chunks]), dtype=torch.float32)
    source = torch.as_tensor([c.source_id for c in chunks], dtype=torch.long)
    clean_actions = actions[source == 0]
    rrt_actions = actions[source == 1]

    schedule = make_vp_schedule(cfg["diffusion"])
    spectral_cfg = cfg["spectral"]
    clean_stats = fit_clean_spectral_stats(
        clean_actions,
        schedule=schedule,
        kernel_size=int(spectral_cfg.get("envelope_kernel_size", 3)),
        eps=float(spectral_cfg.get("eps", 1e-6)),
        global_band_cutoff_norm=float(spectral_cfg.get("global_band_cutoff_norm", 0.2)),
        snr_tau=float(spectral_cfg.get("snr_tau", 1.0)),
        visibility_temperature=float(spectral_cfg.get("visibility_temperature", 0.5)),
    )
    clean_stats.save(run_dir / "spectral" / "clean_stats.pt")

    method = cfg["method"]
    if method in {"sr_tmin", "sr_full"} | SR_FREQMASK_METHODS:
        tmin, bad_visible = annotate_sr_tmin(
            chunks,
            clean_stats,
            schedule,
            eps=float(spectral_cfg.get("eps", 1e-6)),
            kernel_size=int(spectral_cfg.get("envelope_kernel_size", 3)),
            bad_residual_margin=float(spectral_cfg.get("bad_residual_margin", 0.0)),
            global_band_cutoff_norm=float(spectral_cfg.get("global_band_cutoff_norm", 0.2)),
            snr_tau=float(spectral_cfg.get("snr_tau", 1.0)),
            visibility_temperature=float(spectral_cfg.get("visibility_temperature", 0.5)),
        )
        (run_dir / "annotations").mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "annotations" / "sr_tmin.npy", tmin)
        np.save(run_dir / "annotations" / "sr_bad_visible.npy", bad_visible)
        usable = sr_tmin_usable_fraction_from_tmins(tmin[source.numpy() == 1], train_steps=schedule.train_steps)
        with (run_dir / "annotations" / "sr_tmin_usable_fraction_at_t.json").open("w", encoding="utf-8") as f:
            json.dump(selected_sr_tmin_table(usable, [0, 5, 10, 18, 25, 50, 75, 99]), f, indent=2)
        plot_sr_tmin_hist(tmin[source.numpy() == 1], run_dir / "figures" / "sr_tmin_hist.png")

    plot_psd_gcs_vs_rrt(
        clean_actions,
        rrt_actions[: min(len(rrt_actions), 2000)],
        run_dir / "figures" / "psd_gcs_vs_rrt.png",
    )
    visible = visibility_mask(
        clean_stats.clean_psd_mean,
        schedule.sigma,
        tau=float(spectral_cfg.get("snr_tau", 1.0)),
        temperature=float(spectral_cfg.get("visibility_temperature", 0.5)),
    )
    plot_visibility(visible, run_dir / "figures" / "visibility_mask_vs_t.png")
    plot_residual_saliency_examples(
        torch.cat([clean_actions[:4], rrt_actions[:4]], dim=0),
        run_dir / "figures" / "residual_saliency_rollouts.png",
    )
    return chunks, clean_stats, schedule


def train(cfg: dict, max_steps: int | None = None) -> Path:
    seed = int(cfg.get("seed", 0))
    _set_seed(seed)
    method = cfg["method"]
    run_dir = Path(cfg.get("run_root", "runs")) / _run_id(method, seed, cfg.get("run_label"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    chunks, clean_stats, schedule = prepare_chunks_and_stats(cfg, run_dir, seed)
    with (run_dir / "schedule_report.json").open("w", encoding="utf-8") as f:
        json.dump(schedule_report(train_steps=schedule.train_steps), f, indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("device", "auto") != "cpu" else "cpu")
    model_cfg = cfg["model"]
    model = TemporalUNet(
        action_dim=int(model_cfg.get("action_dim", 2)),
        obs_dim=int(model_cfg.get("obs_dim", 6)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-6)),
    )
    ema = EMA(model, decay=float(cfg["training"].get("ema_decay", 0.999)))
    sampler = _make_sampler(method, chunks, schedule, cfg, seed)
    ambient_tmin_idx = getattr(sampler, "tmin_idx", None)

    total_steps = int(max_steps or cfg["training"].get("total_steps", 200_000))
    batch_size = int(cfg["training"].get("batch_size", 256))
    log_every = int(cfg["training"].get("log_every", 100))
    save_every = int(cfg["training"].get("save_every", 0))
    losses = []
    spectral_cfg = cfg["spectral"]
    schedule.sigma = schedule.sigma.to(device)
    for step in range(total_steps):
        batch = sampler.sample(batch_size, device=device)
        noised, _ = add_vp_noise(batch.actions, batch.t_idx, schedule)
        ambient_x_tmin = None
        if method == "ambient_scalar_ambient_loss":
            if ambient_tmin_idx is None:
                raise ValueError("ambient_scalar_ambient_loss requires an AmbientScalarSampler")
            q_mask = batch.source_id == 1
            if q_mask.any():
                noised_q, ambient_x_tmin_q = add_vp_ambient_noise(
                    batch.actions[q_mask],
                    batch.t_idx[q_mask],
                    int(ambient_tmin_idx),
                    schedule,
                )
                noised = noised.clone()
                noised[q_mask] = noised_q
                ambient_x_tmin = torch.empty_like(batch.actions)
                ambient_x_tmin[q_mask] = ambient_x_tmin_q
        pred = model(noised, batch.obs, batch.t_idx)

        if method in {"gcs_only", "rrt_only", "cotrain", "ambient_scalar", "ambient_sampler_x0_mse", "sr_tmin"}:
            loss = F.mse_loss(pred, batch.actions)
        elif method == "ambient_scalar_ambient_loss":
            if ambient_x_tmin is None:
                loss = F.mse_loss(pred, batch.actions)
            else:
                p_mask = batch.source_id == 0
                q_mask = batch.source_id == 1
                source_losses = []
                if p_mask.any():
                    source_losses.append(F.mse_loss(pred[p_mask], batch.actions[p_mask]) * int(p_mask.sum().item()))
                if q_mask.any():
                    source_losses.append(
                        vp_ambient_x0_loss(
                            pred[q_mask],
                            noised[q_mask],
                            ambient_x_tmin[q_mask],
                            batch.t_idx[q_mask],
                            int(ambient_tmin_idx),
                            schedule,
                            ambient_buffer=cfg["training"].get("ambient_loss_buffer"),
                        )
                        * int(q_mask.sum().item())
                    )
                loss = sum(source_losses) / batch.actions.shape[0]
        elif method in SR_FREQMASK_METHODS:
            mask_mode = str(spectral_cfg.get("mask_mode", SR_FREQMASK_METHOD_TO_MASK_MODE[method]))
            loss = sr_freqmask_loss(
                pred,
                batch.actions,
                batch.source_id,
                batch.t_idx,
                clean_stats,
                schedule,
                compat_temperature=float(spectral_cfg.get("compat_temperature", 0.25)),
                low_freq_floor=float(spectral_cfg.get("low_freq_floor", 0.25)),
                global_band_cutoff_norm=float(spectral_cfg.get("global_band_cutoff_norm", 0.2)),
                bad_residual_margin=float(spectral_cfg.get("bad_residual_margin", 0.0)),
                snr_tau=float(spectral_cfg.get("snr_tau", 1.0)),
                visibility_temperature=float(spectral_cfg.get("visibility_temperature", 0.5)),
                kernel_size=int(spectral_cfg.get("envelope_kernel_size", 3)),
                mask_mode=mask_mode,
                eps=float(spectral_cfg.get("eps", 1e-6)),
            )
        elif method == "sr_full":
            loss = sr_full_loss(
                pred,
                batch.actions,
                batch.source_id,
                batch.t_idx,
                clean_stats,
                schedule,
                beta_clean_saliency=float(spectral_cfg.get("beta_clean_saliency", 0.25)),
                saliency_clip=float(spectral_cfg.get("saliency_clip", 5.0)),
                lambda_p_freq=float(spectral_cfg.get("lambda_p_freq", 0.1)),
                q_loss_weight=float(spectral_cfg.get("q_loss_weight", 1.0)),
                compat_temperature=float(spectral_cfg.get("compat_temperature", 0.25)),
                low_freq_floor=float(spectral_cfg.get("low_freq_floor", 0.25)),
                global_band_cutoff_norm=float(spectral_cfg.get("global_band_cutoff_norm", 0.2)),
                bad_residual_margin=float(spectral_cfg.get("bad_residual_margin", 0.0)),
                snr_tau=float(spectral_cfg.get("snr_tau", 1.0)),
                visibility_temperature=float(spectral_cfg.get("visibility_temperature", 0.5)),
                kernel_size=int(spectral_cfg.get("envelope_kernel_size", 3)),
                eps=float(spectral_cfg.get("eps", 1e-6)),
            )
        else:
            raise ValueError(f"unknown method: {method}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip_norm", 1.0)))
        optimizer.step()
        ema.update(model)

        losses.append(float(loss.detach().cpu()))
        if (step + 1) % log_every == 0 or step == total_steps - 1:
            print(f"step={step + 1} loss={np.mean(losses[-log_every:]):.6f} device={device}")
        if save_every > 0 and (step + 1) % save_every == 0 and (step + 1) < total_steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.model.state_dict(),
                    "config": cfg,
                    "losses": losses,
                    "step": step + 1,
                },
                run_dir / f"checkpoint_step_{step + 1}.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "config": cfg,
            "losses": losses,
            "step": total_steps,
        },
        run_dir / "checkpoint_last.pt",
    )
    np.savetxt(run_dir / "losses.txt", np.asarray(losses))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_dir = train(cfg, max_steps=args.max_steps)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
