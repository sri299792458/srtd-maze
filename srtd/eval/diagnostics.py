from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from srtd.data.maze2d_dataset import MazeChunk, build_chunks, load_episodes
from srtd.data.normalization import normalize_xy
from srtd.diffusion.schedules import VPSchedule
from srtd.eval.metrics import average_squared_acceleration, generated_high_freq_residual_energy


def mean_by_source(values: dict[str, list[float]]) -> dict[str, float]:
    return {
        source: float(np.mean(source_values)) if source_values else float("nan")
        for source, source_values in values.items()
    }


def episode_smoothness_by_source(episodes, dt: float = 0.1) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        values[episode.source].append(
            average_squared_acceleration(episode.positions, dt=dt)
        )
    return mean_by_source(values)


def episode_residual_energy_by_source(episodes) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        values[episode.source].append(
            generated_high_freq_residual_energy(normalize_xy(episode.positions))
        )
    return mean_by_source(values)


def sr_tmin_usable_fraction_by_t(
    chunks: list[MazeChunk],
    train_steps: int,
) -> np.ndarray:
    q_tmins = np.asarray(
        [
            train_steps if chunk.sr_tmin_idx is None else chunk.sr_tmin_idx
            for chunk in chunks
            if chunk.source_id == 1
        ],
        dtype=np.int64,
    )
    if len(q_tmins) == 0:
        return np.zeros(train_steps, dtype=np.float32)
    t = np.arange(train_steps, dtype=np.int64)
    return (q_tmins[None, :] <= t[:, None]).mean(axis=1).astype(np.float32)


def fallback_data_sanity(episodes, dt: float = 0.1) -> dict:
    smoothness = episode_smoothness_by_source(episodes, dt=dt)
    residual = episode_residual_energy_by_source(episodes)
    return {
        "smoothness_by_source": smoothness,
        "residual_energy_by_source": residual,
        "rrt_less_smooth_than_clean": bool(
            smoothness.get("q", float("nan")) > smoothness.get("p", float("nan"))
        ),
        "rrt_more_residual_than_clean": bool(
            residual.get("q", float("nan")) > residual.get("p", float("nan"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--policy-horizon", type=int, default=16)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=100)
    args = parser.parse_args()

    episodes = load_episodes(args.episodes)
    report = fallback_data_sanity(episodes)
    chunks = build_chunks(
        episodes,
        policy_horizon=args.policy_horizon,
        obs_horizon=args.obs_horizon,
        stride=args.stride,
    )
    schedule = VPSchedule.cosine(train_steps=args.train_steps)
    report["num_chunks"] = len(chunks)
    report["num_episodes"] = len(episodes)
    report["train_steps"] = schedule.train_steps
    text = json.dumps(report, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

